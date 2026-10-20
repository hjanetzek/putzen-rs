use globset::{Glob, GlobSet, GlobSetBuilder};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::{CleanupRule, Folder};

/// Folder names that are commonly used for out-of-source builds.
const BUILD_DIR_NAMES: &[&str] = &[
    "build",
    "Build",
    "build-*",
    "_build",
    "builddir",
    "cmake-build-*",
    "out",
    "obj",
    "[Dd]ebug",
    "[Dd]ebug-*",
    "[Rr]elease",
    "[Rr]elease-*",
];

/// Files or folders that build systems place at the top of their build folder.
const BUILD_DIR_MARKERS: &[&str] = &[
    "CMakeCache.txt",
    "CMakeFiles",
    "build.ninja",
    ".ninja_log",
    "config.status",
    "meson-info",
    // gradle / android
    "intermediates",
];

/// Compiler outputs. `*.obj` is left out on purpose, it's mostly Wavefront meshes.
const ARTEFACTS: &[&str] = &["*.{o,a,so,lo,la}", "*.so.*"];

/// Upper bound of directory entries looked at when searching for artefacts.
const MAX_ENTRIES_TO_SCAN: usize = 50_000;

enum Evidence {
    Marker(&'static str),
    Artefact(PathBuf),
}

/// Detects build folders without a known "File" -> "Folder" rule.
///
/// A folder is considered a build folder if
/// - its name looks like a build folder (e.g. `build`, `out`, `Release-x64`),
/// - it contains a build system marker (e.g. `CMakeCache.txt`) or compiler outputs (`*.o`, `*.a`, `*.so`, ...),
/// - and none of its files are tracked by git, so vendored prebuilt libraries or
///   checked-in project files are never touched.
pub struct BuildDirHeuristic {
    names: GlobSet,
    artefacts: GlobSet,
}

impl Default for BuildDirHeuristic {
    fn default() -> Self {
        Self {
            names: glob_set(BUILD_DIR_NAMES),
            artefacts: glob_set(ARTEFACTS),
        }
    }
}

fn glob_set(patterns: &[&str]) -> GlobSet {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(Glob::new(pattern).expect("static glob must parse"));
    }
    builder.build().expect("static glob set must build")
}

impl BuildDirHeuristic {
    fn find_evidence(&self, dir: &Path) -> Option<Evidence> {
        if let Some(marker) = BUILD_DIR_MARKERS.iter().find(|m| dir.join(m).exists()) {
            return Some(Evidence::Marker(marker));
        }
        self.find_artefact(dir).map(Evidence::Artefact)
    }

    /// breadth first search for the first compiler output, without following symlinks
    fn find_artefact(&self, dir: &Path) -> Option<PathBuf> {
        let mut queue = VecDeque::from([dir.to_path_buf()]);
        let mut budget = MAX_ENTRIES_TO_SCAN;
        while let Some(current) = queue.pop_front() {
            let Ok(entries) = fs::read_dir(&current) else {
                continue;
            };
            for entry in entries.flatten() {
                if budget == 0 {
                    return None;
                }
                budget -= 1;
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_dir() {
                    queue.push_back(entry.path());
                } else if file_type.is_file() && self.artefacts.is_match(entry.file_name()) {
                    return Some(entry.path());
                }
            }
        }
        None
    }
}

/// Returns true if any file below `dir` is tracked by git.
/// If that cannot be determined inside a git repository, it errs on the safe side.
fn is_tracked_by_git(dir: &Path) -> bool {
    if !dir.ancestors().any(|a| a.join(".git").exists()) {
        return false;
    }
    match Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["ls-files", "-z", "--", "."])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
    {
        Ok(output) if output.status.success() => !output.stdout.is_empty(),
        _ => true,
    }
}

impl CleanupRule for BuildDirHeuristic {
    fn match_folder(&self, folder: &Folder) -> Option<String> {
        let dir = folder.as_ref();
        if !dir.file_name().is_some_and(|n| self.names.is_match(n)) {
            return None;
        }
        let evidence = self.find_evidence(dir)?;
        if is_tracked_by_git(dir) {
            return None;
        }
        Some(match evidence {
            Evidence::Marker(marker) => format!("it contains {marker}"),
            Evidence::Artefact(path) => format!(
                "it contains {}",
                path.strip_prefix(dir).unwrap_or(&path).display()
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{create_dir_all, File};

    fn folder(path: &Path) -> Folder {
        Folder::try_from(path.to_path_buf()).unwrap()
    }

    fn git(dir: &Path, args: &[&str]) -> bool {
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    #[test]
    fn matches_build_dir_with_marker() {
        let root = tempfile::TempDir::new().unwrap();
        let build = root.path().join("Build");
        create_dir_all(&build).unwrap();
        File::create(build.join("CMakeCache.txt")).unwrap();

        let reason = BuildDirHeuristic::default().match_folder(&folder(&build));
        assert_eq!(reason.as_deref(), Some("it contains CMakeCache.txt"));
    }

    #[test]
    fn matches_build_dir_with_nested_artefact() {
        let root = tempfile::TempDir::new().unwrap();
        let release = root.path().join("Release-x64");
        create_dir_all(release.join("lib")).unwrap();
        File::create(release.join("lib").join("libfoo.so.1")).unwrap();

        let reason = BuildDirHeuristic::default().match_folder(&folder(&release));
        assert_eq!(reason.as_deref(), Some("it contains lib/libfoo.so.1"));
    }

    #[test]
    fn ignores_build_dir_without_evidence() {
        let root = tempfile::TempDir::new().unwrap();
        let out = root.path().join("out");
        create_dir_all(&out).unwrap();
        File::create(out.join("mesh.obj")).unwrap();
        File::create(out.join("README.md")).unwrap();

        assert!(BuildDirHeuristic::default()
            .match_folder(&folder(&out))
            .is_none());
    }

    #[test]
    fn ignores_unknown_folder_names() {
        let root = tempfile::TempDir::new().unwrap();
        let src = root.path().join("src");
        create_dir_all(&src).unwrap();
        File::create(src.join("main.o")).unwrap();

        assert!(BuildDirHeuristic::default()
            .match_folder(&folder(&src))
            .is_none());
    }

    #[test]
    fn ignores_build_dir_with_tracked_files() {
        let root = tempfile::TempDir::new().unwrap();
        if !git(root.path(), &["init", "-q"]) {
            eprintln!("git not available, skipping");
            return;
        }
        // vendored prebuilt library, checked into git
        let release = root.path().join("other_lib").join("Release");
        create_dir_all(&release).unwrap();
        File::create(release.join("libktx.a")).unwrap();
        assert!(git(root.path(), &["add", "."]));

        // untracked build output in the same repository
        let build = root.path().join("build");
        create_dir_all(&build).unwrap();
        File::create(build.join("main.o")).unwrap();

        let rule = BuildDirHeuristic::default();
        assert!(rule.match_folder(&folder(&release)).is_none());
        assert!(rule.match_folder(&folder(&build)).is_some());
    }
}
