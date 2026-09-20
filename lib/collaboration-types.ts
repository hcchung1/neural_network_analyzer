export type ViewState = { token: number; layer: number };
export type ModelSelection = { checkpoint_path: string; manifest_spec?: Record<string, unknown>; device: string };
export type SharedAnalysis = {
  model?: ModelSelection;
  modelName?: string;
  inputSchema?: { seq_len: number; feature_dim: number };
  capabilities?: Record<string, unknown>;
  sample?: Record<string, unknown>;
  tokenCount: number;
  maxLayer: number;
  attention: Record<number, number[][]>;
  output: number[];
};
export type RoomEntry = {
  id: string; author: string; text: string; kind: "chat" | "annotation";
  view: ViewState; analysisRevision: number; createdAt: number;
};
export type RoomSnapshot = {
  id: string; revision: number; hostId: string; view: ViewState;
  analysisRevision: number; analysis: SharedAnalysis;
  members: { id: string; name: string; online: boolean }[];
  entries: RoomEntry[]; busy: boolean;
};
export type RoomIdentity = { roomId: string; memberId: string; token: string };
