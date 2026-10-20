use std::collections::HashSet;
use std::convert::TryFrom;
use std::ffi::OsStr;
use std::io::Result;
use std::path::PathBuf;
use std::sync::Arc;

use argh::FromArgs;
use globset::{Glob, GlobSet, GlobSetBuilder};
use jwalk::Parallelism;

use putzen_cli::caches;
use putzen_cli::{
    BuildDirHeuristic, CleanupRule, DecisionContext, DoCleanUp, DryRunCleaner, FileToFolderMatch,
    Folder, FolderProcessed, HumanReadable, NiceInteractiveDecider, NoOpObserver, ProperCleaner,
    RunObserver,
};

#[cfg(feature = "highscore-board")]
use putzen_cli::HighscoreObserver;

/// Static glob pattern used when neither `-a` nor `--hidden` is given.
const DEFAULT_HIDDEN_GLOB: &str = ".worktrees";
/// Static glob pattern used for `-a` / `--dive-into-hidden-folders`.
const ALL_HIDDEN_GLOB: &str = "*";

/// Parse a single glob pattern. Returns a stringified error including the
/// offending input so CLI users see what they typed.
fn parse_glob(s: &str) -> std::result::Result<Glob, String> {
    Glob::new(s).map_err(|e| format!("invalid glob `{s}`: {e}"))
}

/// Cheap first-char heuristic: a glob can only match a hidden basename
/// (which always starts with `.`) if its pattern starts with `.` or with a
/// metacharacter that could expand to one (`*`, `?`, `[`, `{`). Anything
/// else — e.g. `worktrees` — is guaranteed to never match.
fn pattern_can_match_hidden(pattern: &str) -> bool {
    matches!(pattern.chars().next(), Some('.' | '*' | '?' | '[' | '{'))
}

/// Decides whether a hidden directory (basename starts with `.`) is allowed
/// to be entered by the walker. Built once per run from CLI args.
struct HiddenPolicy {
    no_hidden: bool,
    matcher: GlobSet,
}

impl HiddenPolicy {
    fn new(no_hidden: bool, globs: &[Glob]) -> std::result::Result<Self, String> {
        let mut b = GlobSetBuilder::new();
        for g in globs {
            b.add(g.clone());
        }
        let matcher = b
            .build()
            .map_err(|e| format!("failed to build glob set: {e}"))?;
        Ok(Self { no_hidden, matcher })
    }

    /// Should the walker enter this hidden basename?
    /// Caller is responsible for only passing hidden names (starting with `.`).
    fn allows_hidden(&self, name: &OsStr) -> bool {
        if self.no_hidden {
            return false;
        }
        self.matcher.is_match(name)
    }

    /// Build a policy from parsed CLI args. Enforces the mutual-exclusion
    /// rules and applies the default `.worktrees` pattern when no
    /// `--hidden` and no `-a` was passed.
    fn from_args(args: &PutzenCliArgs) -> std::result::Result<Self, String> {
        let dash_a = args.dive_into_hidden_folders;
        let no_hidden = args.no_hidden;
        let hidden_given = !args.hidden.is_empty();

        if no_hidden && hidden_given {
            return Err("`--no-hidden` and `--hidden` are mutually exclusive".into());
        }
        if no_hidden && dash_a {
            return Err("`--no-hidden` and `-a` are mutually exclusive".into());
        }
        if dash_a && hidden_given {
            return Err("`-a` and `--hidden` are mutually exclusive".into());
        }

        if no_hidden {
            return Self::new(true, &[]);
        }

        // Resolve the active set of globs:
        //   -a       -> ["*"]
        //   --hidden -> user list (warn on patterns that can't match a hidden name)
        //   neither  -> [".worktrees"]
        let owned: Vec<Glob> = if dash_a {
            vec![parse_glob(ALL_HIDDEN_GLOB).expect("static glob must parse")]
        } else if hidden_given {
            for g in &args.hidden {
                let pat = g.glob();
                if !pattern_can_match_hidden(pat) {
                    eprintln!(
                        "warning: --hidden `{pat}` does not start with `.`, `*`, `?`, `[`, or `{{` \
                         — hidden basenames always start with `.`, so this pattern will never match"
                    );
                }
            }
            args.hidden.clone()
        } else {
            vec![parse_glob(DEFAULT_HIDDEN_GLOB).expect("static glob must parse")]
        };

        Self::new(false, &owned)
    }
}

/// all supported this to clean up
static FOLDER_TO_CLEANUP: [FileToFolderMatch; 11] = [
    FileToFolderMatch::new("Cargo.toml", "target"),
    FileToFolderMatch::new("package.json", "node_modules"),
    FileToFolderMatch::new("CMakeLists.txt", "build"),
    FileToFolderMatch::new("build.gradle", "build"),
    FileToFolderMatch::new("build.gradle", ".externalNativeBuild"),
    FileToFolderMatch::new("build.gradle", ".cxx"),
    FileToFolderMatch::new("build.gradle.kts", "build"),
    FileToFolderMatch::new("build.gradle.kts", ".externalNativeBuild"),
    FileToFolderMatch::new("build.gradle.kts", ".cxx"),
    FileToFolderMatch::new("Makefile.am", ".libs"),
    FileToFolderMatch::new("Makefile.in", ".libs"),
];

#[derive(FromArgs)]
/// help keeping your disk clean of build and dependency artifacts
///
/// With `--build-dirs`, folders named like build folders (`build`, `out`,
/// `obj`, `Debug`, `Release-x64`, ...) are cleaned as well, if they contain
/// build system files or compiler outputs (`*.o`, `*.a`, `*.so`, ...) and
/// nothing in them is tracked by git.
///
/// Hidden directories are normally skipped, except for `.worktrees`
/// (so colocated git worktrees are cleaned alongside the main checkout).
/// Use `--hidden <GLOB>` to override the list, `--no-hidden` to turn it
/// off entirely, or `-a` to descend into every hidden dir.
///
/// Examples:
///     putzen                              # descends into `.worktrees` by default
///     putzen --hidden '.{worktrees,jj}'  # one glob, two hidden dirs
///     putzen --hidden '.work*'            # any hidden dir starting with `.work`
///     putzen -a                           # every hidden dir (== '*')
///     putzen --no-hidden                  # skip all hidden dirs (legacy)
struct PutzenCliArgs {
    /// show the version number
    #[argh(switch, short = 'v')]
    version: bool,

    /// show the stored highscore board and exit
    #[cfg(feature = "highscore-board")]
    #[argh(switch)]
    scores: bool,

    /// dry-run will never delete anything, good for simulations
    #[argh(switch, short = 'd')]
    dry_run: bool,

    /// switch to say yes to all questions
    #[argh(switch, short = 'y')]
    yes_to_all: bool,

    /// follow symbolic links
    #[argh(switch, short = 'L')]
    follow: bool,

    /// include every hidden directory (== --hidden '*')
    #[argh(switch, short = 'a')]
    dive_into_hidden_folders: bool,

    /// skip every hidden directory (overrides the default `.worktrees`)
    #[argh(switch)]
    no_hidden: bool,

    /// also clean folders that look like build folders and contain
    /// compiler outputs, see above
    #[argh(switch)]
    build_dirs: bool,

    /// glob of hidden directories to descend into (repeatable). Match is
    /// against the full basename including the leading dot, e.g.
    /// `.worktrees`, `.{worktrees,jj}`, `.work*`. Default: `.worktrees`.
    #[argh(option, from_str_fn(parse_glob))]
    hidden: Vec<Glob>,

    /// path where to start with disk clean up.
    #[argh(positional, default = "PathBuf::from(\".\")")]
    folder: PathBuf,
}

#[derive(FromArgs)]
/// interactive cleanup of user-level cache directories
struct CachesCliArgs {
    /// scan root (repeatable). When given, REPLACES the built-in defaults.
    #[argh(option)]
    root: Vec<PathBuf>,
    /// caches whose newest file is younger than this are flagged ACTIVE
    #[argh(option)]
    floor: Option<String>,
    /// dry run: never delete, just show what would happen
    #[argh(switch)]
    dry_run: bool,
    /// skip the deletion confirmation modal
    #[argh(switch, short = 'y')]
    yes: bool,
}

fn main() -> Result<()> {
    let raw: Vec<String> = std::env::args().collect();

    // Route to the caches subcommand before letting argh consume positionals.
    if raw.get(1).map(String::as_str) == Some("caches") {
        let rest: Vec<&str> = raw[2..].iter().map(|s| s.as_str()).collect();
        let parsed = match CachesCliArgs::from_args(&["putzen caches"], &rest) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{}", e.output);
                std::process::exit(if e.status.is_ok() { 0 } else { 1 });
            }
        };
        return caches::run(caches::CachesArgs {
            roots: parsed.root,
            floor: parsed.floor,
            dry_run: parsed.dry_run,
            yes: parsed.yes,
        });
    }

    let args: PutzenCliArgs = argh::from_env();
    if args.version {
        println!("{} {}", env!("CARGO_BIN_NAME"), env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    #[cfg(feature = "highscore-board")]
    if args.scores {
        let highscores = putzen_cli::Highscores::load()?;
        println!("{}", putzen_cli::render_board(&highscores));
        return Ok(());
    }
    visit_path(&args)
}

fn visit_path(args: &PutzenCliArgs) -> Result<()> {
    // the heuristic goes last, so the explicit "File" -> "Folder" rules take precedence
    let mut rules: Vec<Box<dyn CleanupRule>> = FOLDER_TO_CLEANUP
        .iter()
        .map(|rule| Box::new(rule) as Box<dyn CleanupRule>)
        .collect();
    if args.build_dirs {
        rules.push(Box::new(BuildDirHeuristic::default()));
    }
    let rules = Arc::new(rules);
    let to_clean = Arc::clone(&rules);
    let mut decider = NiceInteractiveDecider::default();
    let mut amount_cleaned = 0;
    // with `-L` the same folder can be reached via several paths
    let mut seen = HashSet::new();
    let folder = args
        .folder
        .canonicalize()
        .expect("Folder cannot be canonicalized.");
    let ctx = DecisionContext {
        working_dir: folder.clone(),
        is_dry_run: args.dry_run,
        yes_to_all: args.yes_to_all,
    };

    let cleaner: Box<dyn DoCleanUp> = if args.dry_run {
        Box::new(DryRunCleaner)
    } else {
        Box::new(ProperCleaner)
    };

    let mut observer: Box<dyn RunObserver> = if !args.dry_run {
        #[cfg(feature = "highscore-board")]
        {
            Box::new(HighscoreObserver::load()?)
        }
        #[cfg(not(feature = "highscore-board"))]
        {
            Box::new(NoOpObserver)
        }
    } else {
        Box::new(NoOpObserver)
    };

    let hidden_policy = HiddenPolicy::from_args(args)
        .map_err(|msg| std::io::Error::new(std::io::ErrorKind::InvalidInput, msg))?;

    ctx.println(format!("Start cleaning at {}", folder.display()));
    // Hidden entries are never dropped by jwalk itself: hidden artefact folders
    // like `.libs` or `.externalNativeBuild` must be seen by the rules first,
    // `hidden_policy` only decides about descending into the remaining ones.
    for (folder, rule) in jwalk::WalkDirGeneric::<((), Option<(Folder, usize)>)>::new(folder)
        .skip_hidden(false)
        .follow_links(args.follow)
        .parallelism(Parallelism::RayonNewPool(8))
        .process_read_dir(move |depth, _, _, children| {
            // 1. keep only directories; jwalk resolves symlinks in `file_type` only
            // with `-L`, otherwise e.g. a `lib -> .libs` link would match `.libs` twice
            children.retain(|dir_entry_result| {
                dir_entry_result
                    .as_ref()
                    .map(|dir| dir.file_type().is_dir())
                    .unwrap_or(false)
            });

            // 2. build-artefact marking, matched folders are not descended into
            children.iter_mut().for_each(|child| {
                if let Ok(child) = child {
                    if let Ok(folder) = Folder::try_from(child.path()) {
                        if let Some(i) = to_clean
                            .iter()
                            .position(|rule| rule.match_folder(&folder).is_some())
                        {
                            child.client_state = Some((folder, i));
                            child.read_children_path = None;
                        }
                    }
                }
            });

            // 3. drop unmarked hidden children disallowed by the policy.
            // depth=None is the virtual root call (parent of the starting dir);
            // we must NOT filter those children or we'd block the starting dir itself.
            if depth.is_some() {
                children.retain(|dir_entry_result| {
                    let Ok(dir) = dir_entry_result else {
                        return true;
                    };
                    if dir.client_state.is_some() {
                        return true;
                    }
                    let name = dir.file_name();
                    // byte-level check: works for non-UTF-8 names too, and `.` is always ASCII
                    let is_hidden = name.as_encoded_bytes().first() == Some(&b'.');
                    !is_hidden || hidden_policy.allows_hidden(name)
                });
            }
        })
        .into_iter()
        .filter_map(|f| f.ok())
        .filter_map(|f| f.client_state)
    {
        if !seen.insert(folder.as_ref().to_path_buf()) || !folder.as_ref().exists() {
            continue;
        }
        match folder.accept(&ctx, &*rules[rule], &*cleaner, &mut decider, &mut *observer)? {
            FolderProcessed::Abort => return Ok(()),
            FolderProcessed::Cleaned(size) => amount_cleaned += size,
            FolderProcessed::NoRuleMatch | FolderProcessed::Skipped => {}
        }
    }

    if amount_cleaned > 0 {
        ctx.println(format!("Freed: {}", amount_cleaned.as_human_readable()));
    } else {
        ctx.println("No space freed ;-(");
    }

    if let Some(medals) = observer.on_run_complete(amount_cleaned as u64) {
        println!("{medals}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e2e_scenario() {
        let root_folder = tempfile::TempDir::new().unwrap();
        let target_folder = root_folder.path().join("target");
        std::fs::create_dir(&target_folder).unwrap();
        std::fs::File::create(root_folder.path().join("Cargo.toml")).unwrap();

        // create a target folder with one simple file in it
        std::fs::File::create(target_folder.join("some_artefact")).unwrap();

        // create also a node case in the root folder
        let node_modules_folder = root_folder.path().join("node_modules");
        std::fs::create_dir(&node_modules_folder).unwrap();
        std::fs::File::create(root_folder.path().join("package.json")).unwrap();
        std::fs::File::create(node_modules_folder.join("some_artefact")).unwrap();

        // now we create a nested node case inside the root folder
        let second_node_root_folder = root_folder.path().join("bar");
        std::fs::create_dir(&second_node_root_folder).unwrap();
        let nested_node_modules_folder = second_node_root_folder.join("node_modules");
        std::fs::create_dir(&nested_node_modules_folder).unwrap();
        std::fs::File::create(second_node_root_folder.join("package.json")).unwrap();
        std::fs::File::create(nested_node_modules_folder.join("some_artefact")).unwrap();

        let args = PutzenCliArgs {
            version: false,
            #[cfg(feature = "highscore-board")]
            scores: false,
            dry_run: false,
            yes_to_all: true,
            follow: false,
            dive_into_hidden_folders: false,
            no_hidden: false,
            build_dirs: false,
            hidden: Vec::new(),
            folder: root_folder.path().to_path_buf(),
        };

        visit_path(&args).unwrap();

        assert!(!target_folder.exists());
        assert!(!node_modules_folder.exists());
        assert!(!nested_node_modules_folder.exists());

        assert!(root_folder.path().join("Cargo.toml").exists());
        assert!(root_folder.path().join("package.json").exists());
        assert!(second_node_root_folder.join("package.json").exists());
    }

    fn args_for(folder: &std::path::Path, build_dirs: bool) -> PutzenCliArgs {
        PutzenCliArgs {
            version: false,
            #[cfg(feature = "highscore-board")]
            scores: false,
            dry_run: false,
            yes_to_all: true,
            follow: false,
            dive_into_hidden_folders: false,
            no_hidden: false,
            build_dirs,
            hidden: Vec::new(),
            folder: folder.to_path_buf(),
        }
    }

    #[test]
    fn hidden_artefact_folders_are_cleaned() {
        let root = tempfile::TempDir::new().unwrap();

        // android: app/{build.gradle,.externalNativeBuild}
        let app = root.path().join("app");
        let native_build = app.join(".externalNativeBuild");
        std::fs::create_dir_all(&native_build).unwrap();
        std::fs::File::create(app.join("build.gradle")).unwrap();
        std::fs::File::create(native_build.join("libnative.so")).unwrap();

        // libtool: lib/{Makefile.in,.libs}
        let lib = root.path().join("lib");
        let libs = lib.join(".libs");
        std::fs::create_dir_all(&libs).unwrap();
        std::fs::File::create(lib.join("Makefile.in")).unwrap();
        std::fs::File::create(libs.join("liblwgeom.a")).unwrap();

        // hidden folder without a matching rule stays untouched
        let cache = root.path().join(".cache");
        std::fs::create_dir_all(&cache).unwrap();

        visit_path(&args_for(root.path(), false)).unwrap();

        assert!(!native_build.exists());
        assert!(!libs.exists());
        assert!(app.join("build.gradle").exists());
        assert!(cache.exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_artefact_folder_is_cleaned_once() {
        let root = tempfile::TempDir::new().unwrap();
        // libtool projects often ship `lib -> .libs`
        let libs = root.path().join(".libs");
        std::fs::create_dir_all(&libs).unwrap();
        std::fs::File::create(root.path().join("Makefile.am")).unwrap();
        std::fs::File::create(libs.join("libprime_server.so")).unwrap();
        std::os::unix::fs::symlink(".libs/", root.path().join("lib")).unwrap();

        for follow in [false, true] {
            std::fs::create_dir_all(&libs).unwrap();
            let args = PutzenCliArgs {
                follow,
                ..args_for(root.path(), false)
            };
            visit_path(&args).unwrap();
            assert!(!libs.exists(), "follow={follow}");
        }
    }

    #[test]
    fn build_dirs_are_only_cleaned_when_enabled() {
        let root = tempfile::TempDir::new().unwrap();
        let out = root.path().join("v8").join("out");
        std::fs::create_dir_all(out.join("x64.release")).unwrap();
        std::fs::File::create(out.join("x64.release").join("libv8.a")).unwrap();

        visit_path(&args_for(root.path(), false)).unwrap();
        assert!(out.exists(), "heuristic must be opt-in");

        visit_path(&args_for(root.path(), true)).unwrap();
        assert!(!out.exists());
    }

    #[test]
    fn build_dirs_descends_into_tracked_build_folder() {
        let root = tempfile::TempDir::new().unwrap();
        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .arg("-C")
                .arg(root.path())
                .args(args)
                .output()
                .is_ok_and(|o| o.status.success())
        };
        if !git(&["init", "-q"]) {
            eprintln!("git not available, skipping");
            return;
        }

        // like tracy: profiler/build holds tracked project files,
        // the actual build output lives in profiler/build/unix/obj
        let build = root.path().join("profiler").join("build");
        let obj = build.join("unix").join("obj");
        std::fs::create_dir_all(&obj).unwrap();
        std::fs::File::create(build.join("unix").join("Makefile")).unwrap();
        assert!(git(&["add", "."]));
        std::fs::File::create(obj.join("TracyWorker.o")).unwrap();

        visit_path(&args_for(root.path(), true)).unwrap();

        assert!(build.join("unix").join("Makefile").exists());
        assert!(!obj.exists());
    }

    #[test]
    fn parse_glob_accepts_valid_pattern_with_dot() {
        let g = parse_glob(".worktrees").expect("should parse");
        assert_eq!(g.glob(), ".worktrees");
    }

    #[test]
    fn parse_glob_accepts_brace_expansion() {
        let g = parse_glob(".{worktrees,jj}").expect("should parse");
        assert_eq!(g.glob(), ".{worktrees,jj}");
    }

    #[test]
    fn parse_glob_accepts_wildcard() {
        parse_glob("*").expect("should parse wildcard");
        parse_glob(".work*").expect("should parse prefix wildcard");
    }

    #[test]
    fn pattern_can_match_hidden_heuristic() {
        assert!(pattern_can_match_hidden(".worktrees"));
        assert!(pattern_can_match_hidden(".{worktrees,jj}"));
        assert!(pattern_can_match_hidden(".work*"));
        assert!(pattern_can_match_hidden("*"));
        assert!(pattern_can_match_hidden("?orktrees"));
        assert!(pattern_can_match_hidden("[.a]worktrees"));
        assert!(pattern_can_match_hidden("{.a,.b}"));
        // anything not starting with `.` or a metachar cannot match a hidden basename
        assert!(!pattern_can_match_hidden("worktrees"));
        assert!(!pattern_can_match_hidden(""));
    }

    #[test]
    fn parse_glob_rejects_invalid_pattern() {
        let err = parse_glob("[unterminated").expect_err("should fail");
        // error message includes the offending input so users can self-diagnose
        assert!(err.contains("[unterminated"), "got: {err}");
    }

    fn policy(globs: &[&str]) -> HiddenPolicy {
        let compiled: Vec<Glob> = globs.iter().map(|g| parse_glob(g).unwrap()).collect();
        HiddenPolicy::new(false, &compiled).unwrap()
    }

    #[test]
    fn policy_default_worktrees_only() {
        let p = policy(&[".worktrees"]);
        assert!(p.allows_hidden(".worktrees".as_ref()));
        assert!(!p.allows_hidden(".git".as_ref()));
        assert!(!p.allows_hidden(".cache".as_ref()));
    }

    #[test]
    fn policy_brace_expansion_two_dirs() {
        let p = policy(&[".{worktrees,jj}"]);
        assert!(p.allows_hidden(".worktrees".as_ref()));
        assert!(p.allows_hidden(".jj".as_ref()));
        assert!(!p.allows_hidden(".pijul".as_ref()));
    }

    #[test]
    fn policy_star_matches_all_hidden() {
        let p = policy(&["*"]);
        assert!(p.allows_hidden(".worktrees".as_ref()));
        assert!(p.allows_hidden(".git".as_ref()));
        assert!(p.allows_hidden(".cache".as_ref()));
    }

    #[test]
    fn policy_pattern_without_dot_matches_nothing_hidden() {
        // documented footgun: pattern omits the leading dot, so it can
        // never match a hidden basename (which always starts with `.`)
        let p = policy(&["worktrees"]);
        assert!(!p.allows_hidden(".worktrees".as_ref()));
    }

    #[test]
    fn policy_no_hidden_rejects_everything() {
        let compiled: Vec<Glob> = vec![parse_glob(".worktrees").unwrap()];
        let p = HiddenPolicy::new(true, &compiled).unwrap();
        assert!(!p.allows_hidden(".worktrees".as_ref()));
        assert!(!p.allows_hidden(".anything".as_ref()));
    }

    fn args_from(input: &[&str]) -> std::result::Result<PutzenCliArgs, argh::EarlyExit> {
        PutzenCliArgs::from_args(&["putzen"], input)
    }

    #[test]
    fn from_args_default_includes_worktrees_only() {
        let args = args_from(&[]).unwrap();
        let policy = HiddenPolicy::from_args(&args).unwrap();
        assert!(policy.allows_hidden(".worktrees".as_ref()));
        assert!(!policy.allows_hidden(".git".as_ref()));
    }

    #[test]
    fn from_args_dash_a_includes_everything_hidden() {
        let args = args_from(&["-a"]).unwrap();
        let policy = HiddenPolicy::from_args(&args).unwrap();
        assert!(policy.allows_hidden(".worktrees".as_ref()));
        assert!(policy.allows_hidden(".git".as_ref()));
        assert!(policy.allows_hidden(".whatever".as_ref()));
    }

    #[test]
    fn from_args_no_hidden_rejects_everything() {
        let args = args_from(&["--no-hidden"]).unwrap();
        let policy = HiddenPolicy::from_args(&args).unwrap();
        assert!(!policy.allows_hidden(".worktrees".as_ref()));
        assert!(!policy.allows_hidden(".anything".as_ref()));
    }

    #[test]
    fn from_args_hidden_overrides_default() {
        let args = args_from(&["--hidden", ".git"]).unwrap();
        let policy = HiddenPolicy::from_args(&args).unwrap();
        assert!(policy.allows_hidden(".git".as_ref()));
        // default `.worktrees` is replaced, not merged
        assert!(!policy.allows_hidden(".worktrees".as_ref()));
    }

    #[test]
    fn from_args_hidden_repeatable() {
        let args = args_from(&["--hidden", ".git", "--hidden", ".cache"]).unwrap();
        let policy = HiddenPolicy::from_args(&args).unwrap();
        assert!(policy.allows_hidden(".git".as_ref()));
        assert!(policy.allows_hidden(".cache".as_ref()));
        assert!(!policy.allows_hidden(".worktrees".as_ref()));
    }

    #[test]
    fn from_args_brace_expansion_one_flag_two_dirs() {
        let args = args_from(&["--hidden", ".{worktrees,jj}"]).unwrap();
        let policy = HiddenPolicy::from_args(&args).unwrap();
        assert!(policy.allows_hidden(".worktrees".as_ref()));
        assert!(policy.allows_hidden(".jj".as_ref()));
    }

    #[test]
    fn from_args_invalid_glob_errors_at_parse_time() {
        // argh's from_str_fn surfaces the error at argument-parse time
        let result = args_from(&["--hidden", "[bad"]);
        let early_exit = match result {
            Err(e) => e,
            Ok(_) => panic!("expected parse error for invalid glob"),
        };
        let msg = format!("{early_exit:?}");
        assert!(
            msg.contains("[bad"),
            "error should mention offending input: {msg}"
        );
    }

    #[test]
    fn from_args_no_hidden_conflicts_with_hidden() {
        let args = args_from(&["--no-hidden", "--hidden", ".git"]).unwrap();
        let result = HiddenPolicy::from_args(&args);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected conflict error"),
        };
        assert!(err.contains("--no-hidden"), "got: {err}");
        assert!(err.contains("--hidden"), "got: {err}");
    }

    #[test]
    fn from_args_no_hidden_conflicts_with_dash_a() {
        let args = args_from(&["--no-hidden", "-a"]).unwrap();
        let result = HiddenPolicy::from_args(&args);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected conflict error"),
        };
        assert!(err.contains("--no-hidden"), "got: {err}");
        assert!(err.contains("-a"), "got: {err}");
    }

    #[test]
    fn from_args_dash_a_conflicts_with_hidden() {
        let args = args_from(&["-a", "--hidden", ".git"]).unwrap();
        let result = HiddenPolicy::from_args(&args);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected conflict error"),
        };
        assert!(err.contains("-a"), "got: {err}");
        assert!(err.contains("--hidden"), "got: {err}");
    }

    #[test]
    fn default_run_descends_into_dot_worktrees_but_not_dot_git() {
        let root = tempfile::TempDir::new().unwrap();

        // .worktrees/wt1/target  — should be cleaned by default
        let wt_target = root.path().join(".worktrees").join("wt1").join("target");
        std::fs::create_dir_all(&wt_target).unwrap();
        std::fs::File::create(
            root.path()
                .join(".worktrees")
                .join("wt1")
                .join("Cargo.toml"),
        )
        .unwrap();
        std::fs::File::create(wt_target.join("artefact")).unwrap();

        // .git/target  — should NOT be touched (hidden, not in default include set)
        let git_target = root.path().join(".git").join("target");
        std::fs::create_dir_all(&git_target).unwrap();
        std::fs::File::create(root.path().join(".git").join("Cargo.toml")).unwrap();
        std::fs::File::create(git_target.join("artefact")).unwrap();

        let args = PutzenCliArgs {
            version: false,
            #[cfg(feature = "highscore-board")]
            scores: false,
            dry_run: false,
            yes_to_all: true,
            follow: false,
            dive_into_hidden_folders: false,
            no_hidden: false,
            build_dirs: false,
            hidden: Vec::new(),
            folder: root.path().to_path_buf(),
        };

        visit_path(&args).unwrap();

        assert!(
            !wt_target.exists(),
            ".worktrees/wt1/target should have been cleaned"
        );
        assert!(git_target.exists(), ".git/target must NOT be touched");
    }

    #[test]
    fn no_hidden_skips_dot_worktrees() {
        let root = tempfile::TempDir::new().unwrap();
        let wt_target = root.path().join(".worktrees").join("wt1").join("target");
        std::fs::create_dir_all(&wt_target).unwrap();
        std::fs::File::create(
            root.path()
                .join(".worktrees")
                .join("wt1")
                .join("Cargo.toml"),
        )
        .unwrap();
        std::fs::File::create(wt_target.join("artefact")).unwrap();

        let args = PutzenCliArgs {
            version: false,
            #[cfg(feature = "highscore-board")]
            scores: false,
            dry_run: false,
            yes_to_all: true,
            follow: false,
            dive_into_hidden_folders: false,
            no_hidden: true,
            build_dirs: false,
            hidden: Vec::new(),
            folder: root.path().to_path_buf(),
        };

        visit_path(&args).unwrap();

        assert!(
            wt_target.exists(),
            "--no-hidden must leave .worktrees/wt1/target untouched"
        );
    }
}
