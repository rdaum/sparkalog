#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gdlog_root="$repo_root/reference/gdlog"
lfs_batch_url="https://github.com/harp-lab/gdlog.git/info/lfs/objects/batch"

if [[ ! -d "$gdlog_root/data" ]]; then
    echo "missing $gdlog_root/data; clone the GDlog reference repository first" >&2
    exit 1
fi

declare -a relative_paths
if [[ $# -eq 0 ]]; then
    relative_paths=("data/com-dblp/edge.facts")
elif [[ $1 == "--all" ]]; then
    mapfile -t relative_paths < <(
        find "$gdlog_root/data" -type f -print0 |
            while IFS= read -r -d '' file; do
                if [[ $(sed -n '1p' "$file") == "version https://git-lfs.github.com/spec/v1" ]]; then
                    realpath --relative-to="$gdlog_root" "$file"
                fi
            done | sort
    )
else
    for dataset in "$@"; do
        relative_paths+=("data/$dataset/edge.facts")
    done
fi

for relative_path in "${relative_paths[@]}"; do
    target="$gdlog_root/$relative_path"
    if [[ ! -f "$target" ]]; then
        echo "missing GDlog dataset pointer: $target" >&2
        exit 1
    fi
    if [[ $(sed -n '1p' "$target") != "version https://git-lfs.github.com/spec/v1" ]]; then
        echo "already materialized: $relative_path"
        continue
    fi

    oid=$(sed -n '2s/^oid sha256://p' "$target")
    expected_size=$(sed -n '3s/^size //p' "$target")
    if [[ ! $oid =~ ^[0-9a-f]{64}$ || ! $expected_size =~ ^[0-9]+$ ]]; then
        echo "invalid Git LFS pointer: $target" >&2
        exit 1
    fi

    temporary="$target.partial"
    echo "fetching $relative_path ($expected_size bytes)"
    request=$(jq -cn \
        --arg oid "$oid" \
        --argjson size "$expected_size" \
        '{operation:"download", transfers:["basic"], objects:[{oid:$oid,size:$size}]}')
    response=$(curl --silent --show-error \
        --header 'Accept: application/vnd.git-lfs+json' \
        --header 'Content-Type: application/vnd.git-lfs+json' \
        --data-binary "$request" \
        "$lfs_batch_url")
    download_url=$(jq -r '.objects[0].actions.download.href // empty' <<<"$response")
    if [[ -z $download_url ]]; then
        if [[ $relative_path == "data/com-dblp/edge.facts" ]]; then
            echo "GDlog LFS is unavailable; fetching the original graph from SNAP"
            compressed="$target.gz.partial"
            curl --fail --location --retry 3 \
                --output "$compressed" \
                "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"
            gzip --decompress --stdout "$compressed" >"$temporary"
            rm -f "$compressed"
            actual_rows=$(awk '!/^#/ && NF { rows++ } END { print rows+0 }' "$temporary")
            if [[ $actual_rows != 1049866 ]]; then
                rm -f "$temporary"
                echo "SNAP row-count verification failed: expected 1049866, got $actual_rows" >&2
                exit 1
            fi
            snap_oid=$(sha256sum "$temporary")
            snap_oid=${snap_oid%% *}
            if [[ $snap_oid != ad025cd5933e163e3007e214b3554e0a99d735a3133d0a4e34fd001621219231 ]]; then
                rm -f "$temporary"
                echo "SNAP SHA-256 verification failed: $snap_oid" >&2
                exit 1
            fi
            mv "$temporary" "$target"
            echo "verified $relative_path from SNAP ($actual_rows edges, sha256=$snap_oid)"
            continue
        fi
        jq -r '.objects[0].error.message // "Git LFS server returned no download action"' \
            <<<"$response" >&2
        exit 1
    fi
    declare -a download_headers=()
    while IFS= read -r header; do
        download_headers+=(--header "$header")
    done < <(
        jq -r '(.objects[0].actions.download.header // {}) | to_entries[] | "\(.key): \(.value)"' \
            <<<"$response"
    )
    curl --fail --location --retry 3 \
        "${download_headers[@]}" \
        --output "$temporary" \
        "$download_url"

    actual_size=$(stat --format='%s' "$temporary")
    actual_oid=$(sha256sum "$temporary")
    actual_oid=${actual_oid%% *}
    if [[ $actual_size != "$expected_size" || $actual_oid != "$oid" ]]; then
        rm -f "$temporary"
        echo "verification failed for $relative_path" >&2
        echo "expected size=$expected_size sha256=$oid" >&2
        echo "actual   size=$actual_size sha256=$actual_oid" >&2
        exit 1
    fi

    mv "$temporary" "$target"
    echo "verified $relative_path"
done
