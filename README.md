# service-livebook

Livebook as a dependency, run as a 7-service server.

    mix deps.get
    mix run --no-halt -e 'IO.puts ServiceLivebook.start()'

## Why a dependency and not a fork

The first version of this project was a fork of `livebook-dev/livebook`, checked
out at this path. That is now archived at `weftspun/service-livebook-upstream-fork`.

A fork carries every file upstream has, and the whole of it is ours to keep
current whether or not we change any of it. Nothing here needed a change to
Livebook itself. What the fork bought, and what this drops, is the ability to
patch Livebook in place; what it cost was a second copy of a codebase this
project does not edit, moving on somebody else's schedule.

`{:livebook, "~> 0.19.9"}` states the same thing in one line, and `mix.lock`
records the exact build. If a patch is needed later, a dep can point at a branch
without this project changing shape.

## Known advisories in the pinned tree

Livebook 0.19.9 pins its dependencies exactly, so these come with it and cannot
be raised here without an override that contradicts the pin:

| package | version | advisory          | severity |
| ------- | ------- | ----------------- | -------- |
| req     | 0.5.8   | EEF-CVE-2026-49755 | HIGH     |
| req     | 0.5.8   | EEF-CVE-2026-49756 | LOW      |
| bandit  | 1.11.1  | EEF-CVE-2026-74836 | HIGH     |
| bandit  | 1.11.1  | EEF-CVE-2026-65623 | HIGH     |

`mix hex.audit` reproduces the list. This is recorded rather than silently
accepted: the server binds to loopback by default, and the two HIGH entries are
denial of service against a listening endpoint rather than data exposure.

## Two ways to run it, and one that is closed

    mix run --no-halt -e 'IO.puts ServiceLivebook.start()'   # on the desk, GPU reachable
    docker run -p 8080:8080 weftspun/service-livebook:dev    # container
    sh scripts/run-bwrap.sh                                  # same image, bubblewrap, Linux only

The container and the sandbox run the same artifact: `scripts/run-bwrap.sh`
exports the image's root filesystem once and points bubblewrap at it, so a
difference between the two is a difference in the sandbox and not in the build.
Both were checked by asking `/public/health` for a 200 rather than by watching
the log say it started.

**A Windows service is closed, and this records why rather than leaving it to be
found again.** A Mix release installs one through `erlsrv`, and its `install`
command refuses to run unless `RELEASE_DISTRIBUTION` names a node. Livebook
starts distribution itself, with longnames and its own EPMD module, so a release
that has already started a node makes `Livebook.Application.start_distribution!`
abort with `{:already_started, _}`. That is the measured failure, not a
prediction: the first container did exactly this until `rel/env.sh.eex` set
`RELEASE_DISTRIBUTION=none`. The two settings cannot both hold, so the service
route needs a supervisor that runs the release as a plain foreground process,
such as NSSM, rather than `erlsrv`.

## GPU

The container can reach the card (`docker run --gpus all`, verified), but the
notebooks reach their models through the `pixi` environments in
`6-datasource/anny-render-corpus`, which are on the desk. Until that repository
is bind-mounted, the GPU notebooks run under the first command above.
