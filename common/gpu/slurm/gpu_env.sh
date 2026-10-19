# Source from every sbatch that runs a GPU/Vulkan user (ollama, SAPIEN, Habitat...).
# Keeps the job on the GPUs Slurm allocated:
#   source scripts/slurm/gpu_env.sh      (GPU_AUDIT_LOG=<file> optional)
#
# This cluster's Slurm does not isolate devices (TaskPlugin=task/affinity, no cgroup;
# /dev/nvidia0..7 are rw for everyone): CUDA_VISIBLE_DEVICES is only advice, and Vulkan
# ignores it. So the whole batch script is re-executed inside a user+mount namespace
# (no root needed) where every non-allocated /dev/nvidiaN and its driver entry do not
# exist: no API (CUDA, Vulkan, NVML) can open another GPU. It fails closed if masking fails.
# Inside, CUDA_VISIBLE_DEVICES holds UUIDs; the Slurm indices are in SLURM_GPU_IDX.
# A runtime auditor (scripts/gpu_audit.py), started OUTSIDE the jail, is a second net.
if [[ -n "${SLURM_JOB_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "FATAL: no GPU allocated (missing #SBATCH --gres=gpu:N); a GPU-touching job without gres can land on someone else's GPU" >&2
    exit 4
fi
# ollama also enumerates GPUs through Vulkan, which ignores CUDA_VISIBLE_DEVICES.
export OLLAMA_VULKAN=0 GGML_VK_VISIBLE_DEVICES=-1
# make CUDA indices match nvidia-smi / Slurm (PCI order)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU_ISOLATION=container: for jobs whose GPU user is an Apptainer container started with
# --nvccli (Isaac Sim): Apptainer cannot see the GPU from inside the user namespace of the
# jail, and nvidia-container-cli then exposes ONLY the devices named in
# NVIDIA_VISIBLE_DEVICES. Those are given as UUIDs resolved from the allocated device minors
# (an index there was measured to select a different GPU on the OVX node). The auditor still
# runs; it fails closed if a UUID cannot be resolved. Exported: NVIDIA_VISIBLE_DEVICES,
# GPU_CONTAINER_ISOLATION=1, SLURM_GPU_IDX. Nothing outside the container may touch the GPU.
if [[ -n "${SLURM_JOB_ID:-}" && "${GPU_ISOLATION:-jail}" == container ]]; then
    _ok() { [[ ",${CUDA_VISIBLE_DEVICES}," == *",$1,"* ]]; }
    _uuids=""
    for _info in /proc/driver/nvidia/gpus/*/information; do
        _minor=$(awk -F: '/Device Minor/ {gsub(/[ \t]/, "", $2); print $2}' "$_info")
        _ok "$_minor" || continue
        _uuids+="${_uuids:+,}$(awk -F: '/GPU UUID/ {gsub(/[ \t]/, "", $2); print $2}' "$_info")"
    done
    [[ $(tr ',' '\n' <<< "$_uuids" | grep -c .) == $(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .) ]] \
        || { echo "FATAL: gpu container isolation: cannot resolve UUIDs of ${CUDA_VISIBLE_DEVICES}" >&2; exit 5; }
    if command -v nvidia-smi > /dev/null 2>&1; then
        python3 "$(dirname "${BASH_SOURCE[0]}")/../gpu_audit.py" --root $$ --watch 5 --kill \
            >> "${GPU_AUDIT_LOG:-/dev/null}" 2>&1 &
    fi
    export SLURM_GPU_IDX=$CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES=$_uuids GPU_CONTAINER_ISOLATION=1
    echo "gpu_isolation=container: only GPU(s) ${SLURM_GPU_IDX} (${_uuids}) via nvidia-container-cli"
    return 0 2> /dev/null || exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" && "${GPU_JAIL_INNER:-}" != 1 ]]; then
    if command -v nvidia-smi > /dev/null 2>&1; then
        python3 "$(dirname "${BASH_SOURCE[0]}")/../gpu_audit.py" --root $$ --watch 5 --kill \
            >> "${GPU_AUDIT_LOG:-/dev/null}" 2>&1 &
    fi
    # exec keeps this PID, so the auditor root and Slurm's signals (SIGUSR1) still apply
    exec env GPU_JAIL_INNER=1 unshare --map-root-user --mount --propagation private \
        /bin/bash "$0" "$@"
fi

if [[ "${GPU_JAIL_INNER:-}" == 1 ]]; then
    # Same layout nvidia-container gives a container: the other GPUs' device nodes and
    # driver entries do not exist (a /dev/null stand-in breaks CUDA init), and the driver
    # may not recreate nodes (ModifyDeviceFiles: 0). Slurm IDX == /dev/nvidiaN minor.
    _jail_fail() { echo "FATAL: gpu_jail: $*" >&2; exit 5; }
    _jail_ok() { [[ ",${CUDA_VISIBLE_DEVICES}," == *",$1,"* ]]; }
    _jail=$(mktemp -d "${TMPDIR:-/tmp}/gpujail.XXXXXX") || _jail_fail "mktemp"
    mount -t tmpfs -o mode=755 tmpfs "$_jail" || _jail_fail "tmpfs"
    mkdir "$_jail/dev" "$_jail/gpus"
    _uuids=""
    for _info in /proc/driver/nvidia/gpus/*/information; do
        _bus=$(basename "$(dirname "$_info")")
        _minor=$(awk -F: '/Device Minor/ {gsub(/[ \t]/, "", $2); print $2}' "$_info")
        _jail_ok "$_minor" || continue
        mkdir "$_jail/gpus/$_bus"
        mount --bind "/proc/driver/nvidia/gpus/$_bus" "$_jail/gpus/$_bus" || _jail_fail "bind $_bus"
        _uuids+="${_uuids:+,}$(awk -F: '/GPU UUID/ {gsub(/[ \t]/, "", $2); print $2}' "$_info")"
    done
    sed 's/^ModifyDeviceFiles: .*/ModifyDeviceFiles: 0/' /proc/driver/nvidia/params > "$_jail/params"
    for _e in /dev/* /dev/.[!.]*; do
        [[ -e "$_e" || -L "$_e" ]] || continue
        _n=${_e#/dev/}
        if [[ $_n =~ ^nvidia([0-9]+)$ ]] && ! _jail_ok "${BASH_REMATCH[1]}"; then continue; fi
        if [[ -L "$_e" ]]; then cp -P "$_e" "$_jail/dev/$_n"
        elif [[ -d "$_e" ]]; then mkdir "$_jail/dev/$_n" && mount --rbind "$_e" "$_jail/dev/$_n" 2> /dev/null
        else touch "$_jail/dev/$_n" && mount --bind "$_e" "$_jail/dev/$_n" 2> /dev/null; fi
    done
    mount --rbind "$_jail/dev" /dev || _jail_fail "bind /dev"
    mount --bind "$_jail/gpus" /proc/driver/nvidia/gpus || _jail_fail "bind gpus"
    mount --bind "$_jail/params" /proc/driver/nvidia/params || _jail_fail "bind params"
    # verify (fail closed): no other GPU node or driver entry is left
    for _dev in /dev/nvidia[0-9]*; do
        _jail_ok "${_dev#/dev/nvidia}" || _jail_fail "$_dev still present"
    done
    [[ $(ls /proc/driver/nvidia/gpus | wc -l) == $(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l) ]] \
        || _jail_fail "driver still lists other GPUs"
    [[ -n $_uuids ]] || _jail_fail "no allocated GPU found"
    # indices change once only the allocated GPUs exist; UUIDs do not
    export SLURM_GPU_IDX=$CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES=$_uuids
    echo "gpu_jail: only GPU(s) ${SLURM_GPU_IDX} reachable (${_uuids})"
fi
