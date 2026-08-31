TOPAS is not installed locally: it runs on the remote machine '{{remote_host}}' over SSH. Always
use `run_topas_refinement_remote` to run refinements, never `run_topas_refinement`. That tool
refuses to start if TOPAS is already busy on the remote machine; you can check that yourself with
`check_remote_topas_running`. Each remote run gets its own run directory which is copied back
locally, so the paths in the result refer to the local copies. TOPAS runs with that directory as
its working directory, so every filename in the .inp file — the data files it reads and the names
given to the Out_* macros — should be a bare filename with no directory part.
