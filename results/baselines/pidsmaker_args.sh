#!/bin/bash
# Hyperparameters exactly as PIDSMaker's docs/docs/tuned_systems.md (commit 2289cd9). Systems or
# datasets without a tuned entry there (MAGIC, ThreaTrace, original ORTHRUS, and every system on
# ATLASV2_EDR) run their config/<system>.yml defaults, i.e. the original papers' hyperparameters.
tuned_args() {
  local sys=$1 ds=$2
  case "$sys/$ds" in
    orthrus_non_snooped/CADETS_E3)     echo "--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256" ;;
    orthrus_non_snooped/THEIA_E3)      echo "--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256 --construction.time_window_size=5" ;;
    orthrus_non_snooped/CLEARSCOPE_E3) echo "--training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1" ;;
    # OpTC + TGN systems: neighbour size 20 -> 10 only because the documented setting needs >251 GB
    # RAM in "Computing TGN last neighbor graphs" (ORTHRUS-ns: 253 GB anon RSS, kernel OOM kill)
    orthrus_non_snooped/optc_h501)     echo "--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256 --batching.intra_graph_batching.tgn_last_neighbor.tgn_neighbor_size=10" ;;
    orthrus/optc_h501)                 echo "--batching.intra_graph_batching.tgn_last_neighbor.tgn_neighbor_size=10" ;;
    kairos/CADETS_E3)                  echo "--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256" ;;
    # documented batch 1024 kept: the 24 GB CUDA OOM was the eval-mode TGN memory flush, now chunked
    # in pidsmaker/tgn.py (same result); 512 did not avoid it
    kairos/THEIA_E3)                   echo "--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256" ;;
    kairos/CLEARSCOPE_E3)              echo "--training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1" ;;
    kairos/optc_h501)                  echo "--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=16 --batching.intra_graph_batching.tgn_last_neighbor.tgn_neighbor_size=10" ;;
    velox/CADETS_E3)                   echo "--training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256" ;;
    velox/THEIA_E3)                    echo "--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256" ;;
    velox/CLEARSCOPE_E3)               echo "--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1" ;;
    velox/optc_h501)                   echo "--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256" ;;
    nodlink/CADETS_E3)                 echo "--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=128 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods=none" ;;
    nodlink/THEIA_E3)                  echo "--training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=128 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods=none" ;;
    nodlink/CLEARSCOPE_E3)             echo "--training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1" ;;
    nodlink/optc_h501)                 echo "--training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256" ;;
    reflex_ts/*|reflex_ts2/*|reflex_ts3/*) tuned_args orthrus_non_snooped "$ds" ;;   # same documented setting
    # DEFAULT-config runs (validation plan P2.5 / P2.7): no tuned hyperparameter; only the OpTC TGN
    # neighbour size stays at 10 (hardware: the default 20 needs > 251 GB RAM)
    reflex_tsd/optc_h501|reflex_tsd2/optc_h501|reflex_tsd3/optc_h501|orthrus_ns_def/optc_h501|kairos_def/optc_h501)
                                       echo "--batching.intra_graph_batching.tgn_last_neighbor.tgn_neighbor_size=10" ;;
    reflex_tsd*/*|orthrus_ns_def/*|kairos_def/*|velox_def/*|nodlink_def/*) echo "" ;;
    # mimicry-attack runs (validation plan P2.8): exactly the base system's own setting
    *_mim1000/*|*_mim5000/*)           tuned_args "${sys%_mim*}" "$ds" ;;
    # masquerade runs (validation plan P2.9): exactly the base system's own setting
    *_masq/*)                          tuned_args "${sys%_masq}" "$ds" ;;
    *) echo "" ;;
  esac
}
