#!/usr/bin/env bash
# Our own ODX distribution on the WebODM server: builds ODX from the HikeAndMap forks (ODX, OpenSfM, mvs-texturing)
# and runs it as a WebODM processing node.
#
#   update-node.sh build            pull HikeAndMap/ODX master and build the images (a few hours: everything is compiled);
#                                   the running nodes keep working meanwhile, only slower
#   update-node.sh deploy [image]   run the newest build (or the given image) as its node; refuses while that node has a task
#                                   queued or running
#   update-node.sh status           every node of ours: the ODX commit and patch level it runs and its task count
#
# A node is named after the highest patch it contains: node-patch4 holds [PATCH1] to [PATCH4]. The patch level is the highest
# [PATCHn] console marker in HikeAndMap/ODX (every patch prints one, the ones inside OpenSfM or mvs-texturing are switched on
# from ODX), stored in the image as the label org.hikeandmap.odx.patch. A newer build with the same patch level replaces
# that node; a build with a new patch gets a new node next to the old one, so tasks on the old one can finish - add the new
# one in WebODM and remove the old one there and with "docker rm -f" when it is no longer needed.
#
# A node gets the limits of our server (190 GiB RAM, 690 GiB RAM + swap, 15 CPUs, network webodm_default, restart
# unless-stopped) and keeps finished tasks for 14 days (NodeODX's own default is 2 days), so a task can still be restarted
# from a stage after a run of two days. Override with MEMORY, MEMORY_SWAP, CPUS, CLEANUP_MINUTES, NETWORK, NODE.
# The first time a node exists, add it in WebODM: Processing nodes > Add, hostname = the node's name, port 3000.
set -euo pipefail

WORK=${WORK:-/root/hikeandmap}
NETWORK=${NETWORK:-webodm_default}
MEMORY=${MEMORY:-190g}
MEMORY_SWAP=${MEMORY_SWAP:-690g}
CPUS=${CPUS:-15}
CLEANUP_MINUTES=${CLEANUP_MINUTES:-20160}

ODX_REPO=https://github.com/HikeAndMap/ODX.git
# NodeODX from our fork (WebODM/NodeODX 02296467 + the Dockerfile without BuildKit warnings), at a fixed commit
NODEODX_REPO=https://github.com/HikeAndMap/NodeODX.git
NODEODX_COMMIT=f3a2b7bdfecad44dbb725f892ba280da1ec1136b

LOGS=$WORK/logs

# The highest [PATCHn] marker in an ODX source folder
patch_level_of_source() {
    grep -rhoE '\[PATCH[0-9]+\]' --include='*.py' --include='*.cmake' --include='*.sh' "$1" 2>/dev/null \
        | grep -oE '[0-9]+' | sort -n | tail -1 || true
}

label_of() {
    docker image inspect -f "{{index .Config.Labels \"$2\"}}" "$1" 2>/dev/null | sed 's/<no value>//' || true
}

# The patch level of an image: its label, or - for an image built before the label existed - the build folder's source,
# if the build folder is at the image's commit
patch_level_of_image() {
    local level commit
    level=$(label_of "$1" org.hikeandmap.odx.patch)
    if [ -z "$level" ]; then
        commit=$(label_of "$1" org.hikeandmap.odx.commit)
        if [ -n "$commit" ] && [ -d "$WORK/ODX/.git" ] && [ "$(git -C "$WORK/ODX" rev-parse --short=8 HEAD)" = "$commit" ]; then
            level=$(patch_level_of_source "$WORK/ODX")
        fi
    fi
    echo "$level"
}

node_task_count() {
    # Queued + running tasks, from NodeODX's /info
    docker exec "$1" python3 -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://localhost:3000/info'))['taskQueueCount'])"
}

build() {
    mkdir -p "$WORK" "$LOGS"

    # $WORK/ODX is only a build copy of GitHub: it is reset to HikeAndMap/ODX master every time
    if [ ! -d "$WORK/ODX/.git" ]; then git clone "$ODX_REPO" "$WORK/ODX"; fi
    git -C "$WORK/ODX" remote set-url origin "$ODX_REPO"
    git -C "$WORK/ODX" fetch origin
    git -C "$WORK/ODX" reset --hard origin/master
    local commit level
    commit=$(git -C "$WORK/ODX" rev-parse --short=8 HEAD)
    level=$(patch_level_of_source "$WORK/ODX")
    level=${level:-0}

    if [ ! -d "$WORK/NodeODX/.git" ]; then git clone "$NODEODX_REPO" "$WORK/NodeODX"; fi
    # an older build folder may still point to WebODM/NodeODX
    git -C "$WORK/NodeODX" remote set-url origin "$NODEODX_REPO"
    git -C "$WORK/NodeODX" fetch origin
    git -C "$WORK/NodeODX" checkout --force "$NODEODX_COMMIT"

    echo "Building hikeandmap/odx:$commit, patch level $level (log: $LOGS/odx-$commit.log)"
    docker build --label "org.hikeandmap.odx.commit=$commit" --label "org.hikeandmap.odx.patch=$level" \
        -t "hikeandmap/odx:$commit" "$WORK/ODX" 2>&1 | tee "$LOGS/odx-$commit.log"

    echo "Building hikeandmap/nodeodx:$commit (log: $LOGS/nodeodx-$commit.log)"
    docker build --build-arg "FROM_REPO=hikeandmap/odx:$commit" \
        --label "org.hikeandmap.odx.commit=$commit" --label "org.hikeandmap.odx.patch=$level" \
        -t "hikeandmap/nodeodx:$commit" -t hikeandmap/nodeodx:latest "$WORK/NodeODX" 2>&1 | tee "$LOGS/nodeodx-$commit.log"

    echo "Built hikeandmap/nodeodx:$commit (patch level $level, node node-patch$level). Deploy it with: $0 deploy"
}

deploy() {
    local image=${1:-hikeandmap/nodeodx:latest}
    docker image inspect "$image" > /dev/null

    local level node commit
    level=$(patch_level_of_image "$image")
    if [ -z "$level" ] && [ -z "${NODE:-}" ]; then
        echo "The patch level of $image is not known (it has no org.hikeandmap.odx.patch label and the build folder is at another commit)."
        echo "Rebuild it, or name the node yourself: NODE=node-patchN $0 deploy $image"
        exit 1
    fi
    node=${NODE:-node-patch$level}
    commit=$(label_of "$image" org.hikeandmap.odx.commit)

    if docker container inspect "$node" > /dev/null 2>&1; then
        if [ "$(docker container inspect -f '{{.State.Running}}' "$node")" = "true" ]; then
            local tasks
            tasks=$(node_task_count "$node")
            if [ "$tasks" != "0" ]; then
                echo "$node has $tasks task(s) queued or running: not replaced. Wait until they have finished, or cancel them in WebODM."
                exit 1
            fi
        fi
        echo "Removing $node (the files of its finished tasks go with it: a finished task can no longer be restarted from a stage)"
        docker rm -f "$node"
        local existed=1
    fi

    docker run -d --name "$node" --network "$NETWORK" --restart unless-stopped \
        --label "org.hikeandmap.node=1" \
        --memory "$MEMORY" --memory-swap "$MEMORY_SWAP" --cpus "$CPUS" "$image" --cleanup_tasks_after "$CLEANUP_MINUTES"
    echo "$node runs $image (ODX commit $commit, patch level ${level:-?}); finished tasks are kept $((CLEANUP_MINUTES / 1440)) days."
    if [ -z "${existed:-}" ]; then
        echo "New node: add it in WebODM, Processing nodes > Add, hostname $node, port 3000."
    fi
}

status() {
    local nodes
    nodes=$(docker ps -a --filter "label=org.hikeandmap.node=1" --format '{{.Names}}')
    if [ -z "$nodes" ]; then
        echo "No node of ours exists yet"
        return
    fi
    docker ps -a --filter "label=org.hikeandmap.node=1" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
    for node in $nodes; do
        local image
        image=$(docker container inspect -f '{{.Image}}' "$node") # the image it was started from, even after "latest" moved on
        echo "$node: ODX commit $(label_of "$image" org.hikeandmap.odx.commit), patch level $(patch_level_of_image "$image")"
        if [ "$(docker container inspect -f '{{.State.Running}}' "$node")" = "true" ]; then
            echo "  tasks queued or running: $(node_task_count "$node")"
        fi
    done
}

case "${1:-}" in
    build) build ;;
    deploy) deploy "${2:-}" ;;
    status) status ;;
    *) echo "Usage: $0 build | deploy [image] | status"; exit 1 ;;
esac
