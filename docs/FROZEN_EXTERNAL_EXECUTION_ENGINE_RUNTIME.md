# Final Frozen External Execution Engine Runtime

This milestone implements the complete one-time execution path while keeping both `external_execution_enabled` and `real_wsi_access_enabled` false. The committed and CI-tested modes are preflight and synthetic integration only.

The real branch requires explicit dual enablement plus the exact execution token. It atomically consumes the one permitted execution before the first WSI pixel read, runs unchanged Spatial v2 and v3 sampling, validates 300 coordinates and 300x512 embeddings per view, applies only frozen scalers/models and threshold, writes interruption-safe partial receipts, seals failures, seals completion, and permanently refuses a second execution.

No real external execution occurs in this milestone.
