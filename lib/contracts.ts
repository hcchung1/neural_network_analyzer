import { z } from "zod";

export const SidecarErrorSchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.any()).optional().default({}),
    recovery: z.array(z.string()).optional().default([]),
    evidence: z.array(z.any()).optional().default([]),
  }),
});

export const LoadModelRequestSchema = z.object({
  checkpoint_path: z.string().min(1, "Checkpoint path is required"),
  manifest_spec: z.record(z.any()).optional(),
  requested_phase: z.string().optional(),
  device: z.string().default("cpu"),
});

export const LoadModelResponseSchema = z.object({
  status: z.string(),
  model_name: z.string(),
  phase: z.string().nullable().optional(),
  device: z.string(),
  capabilities: z.record(z.any()),
});

export const CreateSessionRequestSchema = z.object({
  input_data: z.array(z.any()),
  requested_probes: z.array(z.string()).optional().default([]),
  token_idx: z.number().default(0),
});

export const SessionSummarySchema = z.object({
  session_id: z.string(),
  created_at: z.number(),
  model_name: z.string(),
  output_shape: z.array(z.number()),
  output_sample: z.array(z.number()),
  probes_captured: z.array(z.string()),
});
