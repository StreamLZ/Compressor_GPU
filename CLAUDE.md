# Project Rules

## BUILD ENVIRONMENT: nvcc NEEDS vcvarsall
Bare `nvcc` fails with "Cannot find compiler 'cl.exe' in PATH" - cl
is not on the system PATH. `zig build ptx` handles this itself (its
generated script calls vcvarsall first). For MANUAL nvcc runs
(e.g. one-off -lineinfo builds), chain through:
`call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64`
then nvcc in the same cmd session.

## NEVER REVERT WITHOUT PERMISSION
NEVER run `git checkout --`, `git restore`, `git revert`, or any command that undoes file changes without the user explicitly saying "revert", "undo", or "restore". If a change made things slower or didn't help, LEAVE THE CODE AS-IS and report the results. The user decides whether to revert. No exceptions.

## CROSS-BACKEND SHA GATE AFTER ENCODER CHANGES
After ANY change to the CUDA or VK encoder, encode enwik8 at L1/L3/L5
on BOTH backends and verify the frames are byte-identical (SHA256
compare) before claiming the change done. All five levels when the
change touches level-specific paths. The two backends are each
other's oracle; a silent frame divergence is a bug even when both
sides roundtrip their own output.

## NO AI ATTRIBUTION IN COMMITS
Do not append "Co-Authored-By: Claude ..." or any AI attribution
trailer to commit messages.
