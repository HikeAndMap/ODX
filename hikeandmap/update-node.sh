#!/usr/bin/env bash
# Our own ODX distribution on the WebODM server: builds ODX from the HikeAndMap forks (ODX, OpenSfM, mvs-texturing)
# and runs it as a WebODM processing node.
#
#   update-node.sh build     pull HikeAndMap/ODX master and build the images (a few hours: everything is compiled);
#                            the running node keeps working meanwhile, only slower
#   update-node.sh deploy    replace the node with the newest build; refuses while the node has a task queued or running
#   update-node.sh status    the node, the ODX commit it runs and its task count
#
# The node gets the same limits as node-odx-fix-3 (190 GiB RAM, 690 GiB RAM + swap, 15 CPUs, network webodm_default,
# restart unless-stopped). The first time, add it in WebODM: Processing nodes > Add, hostname node-hikeandmap, port 3000.
set -euo pipefail

WORK=${WORK:-/root/hikeandmap}
NODE=${NODE:-node-hikeandmap}
NETWORK=${NETWORK:-webodm_default}
MEMORY=${MEMORY:-190g}
MEMORY_SWAP=${MEMORY_SWAP:-690g}
CPUS=${CPUS:-15}

ODX_REPO=https://github.com/HikeAndMap/ODX.git
# NodeODX is used unchanged, at a fixed commit; it is forked as soon as we change it
NODEODX_REPO=https://github.com/WebODM/NodeODX.git
NODEODX_COMMIT=02296467c0ca2142c2c7771876788bacbed102f6

LOGS=$WORK/logs

node_task_count() {
    # Queued + running tasks, from NodeODX's /info
    docker exec "$NODE" python3 -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://localhost:3000/info'))['taskQueueCount'])"
}

build() {
    mkdir -p "$WORK" "$LOGS"

    # $WORK/ODX is only a build copy of GitHub: it is reset to HikeAndMap/ODX master every time
    if [ ! -d "$WORK/ODX/.git" ]; then git clone "$ODX_REPO" "$WORK/ODX"; fi
    git -C "$WORK/ODX" fetch origin
    git -C "$WORK/ODX" reset --hard origin/master
    local commit
    commit=$(git -C "$WORK/ODX" rev-parse --short=8 HEAD)

    if [ ! -d "$WORK/NodeODX/.git" ]; then git clone "$NODEODX_REPO" "$WORK/NodeODX"; fi
    git -C "$WORK/NodeODX" fetch origin
    git -C "$WORK/NodeODX" checkout --force "$NODEODX_COMMIT"

    echo "Building hikeandmap/odx:$commit (log: $LOGS/odx-$commit.log)"
    docker build --label "org.hikeandmap.odx.commit=$commit" -t "hikeandmap/odx:$commit" "$WORK/ODX" 2>&1 | tee "$LOGS/odx-$commit.log"

    echo "Building hikeandmap/nodeodx:$commit (log: $LOGS/nodeodx-$commit.log)"
    docker build --build-arg "FROM_REPO=hikeandmap/odx:$commit" --label "org.hikeandmap.odx.commit=$commit" \
        -t "hikeandmap/nodeodx:$commit" -t hikeandmap/nodeodx:latest "$WORK/NodeODX" 2>&1 | tee "$LOGS/nodeodx-$commit.log"

    echo "Built hikeandmap/nodeodx:$commit. Deploy it with: $0 deploy"
}

deploy() {
    local image=${1:-hikeandmap/nodeodx:latest}
    docker image inspect "$image" > /dev/null

    if docker container inspect "$NODE" > /dev/null 2>&1; then
        if [ "$(docker container inspect -f '{{.State.Running}}' "$NODE")" = "true" ]; then
            local tasks
            tasks=$(node_task_count)
            if [ "$tasks" != "0" ]; then
                echo "$NODE has $tasks task(s) queued or running: not replaced. Wait until they have finished, or cancel them in WebODM."
                exit 1
            fi
        fi
        echo "Removing $NODE (the files of its finished tasks go with it: a finished task can no longer be restarted from a stage)"
        docker rm -f "$NODE"
    fi

    docker run -d --name "$NODE" --network "$NETWORK" --restart unless-stopped \
        --memory "$MEMORY" --memory-swap "$MEMORY_SWAP" --cpus "$CPUS" "$image"
    echo "$NODE runs $image (ODX commit $(docker image inspect -f '{{index .Config.Labels "org.hikeandmap.odx.commit"}}' "$image"))."
    echo "First time only: add it in WebODM, Processing nodes > Add, hostname $NODE, port 3000."
}

status() {
    if ! docker container inspect "$NODE" > /dev/null 2>&1; then
        echo "$NODE does not exist yet"
        return
    fi
    docker ps -a --filter "name=^/${NODE}$" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
    echo "ODX commit: $(docker container inspect -f '{{index .Config.Labels "org.hikeandmap.odx.commit"}}' "$NODE")"
    if [ "$(docker container inspect -f '{{.State.Running}}' "$NODE")" = "true" ]; then
        echo "Tasks queued or running: $(node_task_count)"
    fi
}

case "${1:-}" in
    build) build ;;
    deploy) deploy "${2:-}" ;;
    status) status ;;
    *) echo "Usage: $0 build | deploy [image] | status"; exit 1 ;;
esac
