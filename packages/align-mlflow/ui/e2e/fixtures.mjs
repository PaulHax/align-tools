import { createHash } from "node:crypto";

const epoch = Date.UTC(2026, 0, 1);
const methods = [
  "Baseline",
  "Direct regression",
  "Fewshot comparative regression",
  "Zeroshot comparative regression",
  "Method E",
  "Method F",
  "Method G",
  "Baseline",
];
const pairs = Array.from({ length: 24 }, (_, index) => ({
  scenario: index < 12 ? "desert-evaluation-3" : "urban-evaluation-3",
  target: `example-${index % 12 < 6 ? "merit" : "affiliation"}-${((index % 6) / 5).toFixed(1)}`,
}));

export const createPopulation = (
  runCount = 8,
  { duplicateScores = false } = {},
) => {
  const configurations = Array.from({ length: runCount }, (_, index) => {
    const number = String(index + 1).padStart(3, "0");
    const adm = methods[index % methods.length];
    return {
      id: (index + 1).toString(16).padStart(32, "0"),
      name: `${adm} configuration ${number} with a complete descriptive evaluation name`,
      adm,
      model: `example/model-${number}-with-a-long-descriptive-checkpoint-name`,
      alignmentMode: adm === "Baseline" ? "unaligned" : "aligned",
      driver: "example.drivers.OpenWorldDriver",
      sourceVersion: "example-producer-version-1",
      sourceFamily: "example-source-family",
      scoringVersion: "example-scoring-version-1",
    };
  });
  const traces = configurations.flatMap((configuration, runIndex) =>
    pairs.map((pair, caseIndex) => {
      const id = `example-session-${runIndex + 1}-${caseIndex + 1}`;
      const traceId = `tr-example-${runIndex + 1}-${caseIndex + 1}`;
      const requested =
        configuration.alignmentMode === "unaligned"
          ? undefined
          : runIndex === 1 && caseIndex === 1
            ? pairs[5].target
            : pair.target;
      const measured = ((runIndex * 17 + caseIndex * 3) % 101) / 100;
      const score =
        duplicateScores && runIndex === 1
          ? caseIndex < 2
            ? 0.5
            : caseIndex === 2
              ? 0.5001
              : measured === 0.5
                ? 0.51
                : measured
          : measured;
      const time = new Date(
        epoch + runIndex * 86_400_000 + caseIndex * 60_000,
      ).toISOString();
      return {
        trace_id: traceId,
        trace_location: {
          type: "MLFLOW_EXPERIMENT",
          mlflow_experiment: { experiment_id: "1" },
        },
        request_time: time,
        execution_duration: "0.1s",
        state: "OK",
        tags: {
          "mlflow.traceName": "01 Evaluate example",
          "align.import_complete": "source-record-v1",
          "align.source_version": configuration.sourceVersion,
          "align.source_family": configuration.sourceFamily,
          "align.alignment_mode": configuration.alignmentMode,
          "align.driver": configuration.driver,
          step: "1",
          episode: String(caseIndex + 1),
          run: configuration.name,
          adm: configuration.adm,
          llm: configuration.model,
          scenario_id: pair.scenario,
          ...(requested ? { alignment_target_id: requested } : {}),
        },
        trace_metadata: {
          "mlflow.trace.session": id,
          "mlflow.sourceRun": configuration.id,
        },
        assessments: [
          {
            assessment_id: `example-assessment-${runIndex + 1}-${caseIndex + 1}`,
            trace_id: traceId,
            assessment_name: "ta3_session_alignment_score",
            source: { source_type: "CODE", source_id: "raw_align_system.log" },
            create_time: time,
            last_update_time: time,
            valid: true,
            feedback: { value: score },
            metadata: {
              "mlflow.trace.session": id,
              alignment_target_id: pair.target,
              scoring_version: configuration.scoringVersion,
            },
          },
        ],
      };
    }),
  );
  const runs = configurations.map((configuration, index) => {
    const values = traces
      .filter(
        (trace) =>
          trace.trace_metadata["mlflow.sourceRun"] === configuration.id,
      )
      .map((trace) => trace.assessments[0].feedback.value);
    const metrics = {
      episode_count: 24,
      scored_episode_count: 24,
      missing_score_count: 0,
      target_mismatch_count: index === 1 ? 1 : 0,
      ta3_alignment_mean:
        values.reduce((sum, value) => sum + value, 0) / values.length,
    };
    return {
      info: {
        run_uuid: configuration.id,
        run_id: configuration.id,
        run_name: configuration.name,
        experiment_id: "1",
        start_time: epoch + index * 86_400_000,
        end_time: epoch + index * 86_400_000 + 1000,
        status: "FINISHED",
        lifecycle_stage: "active",
        artifact_uri: `mlflow-artifacts:/1/${configuration.id}/artifacts`,
      },
      data: {
        params: [],
        tags: [
          { key: "mlflow.runName", value: configuration.name },
          { key: "adm", value: configuration.adm },
        ],
        metrics: Object.entries(metrics).map(([key, value]) => ({
          key,
          value,
          timestamp: epoch,
          step: 0,
        })),
      },
      inputs: { dataset_inputs: [] },
      outputs: { model_outputs: [] },
    };
  });
  return {
    configurations,
    traces,
    runs,
    pairs,
    experiment: {
      experiment_id: "1",
      name: "Example Open World evaluations",
      artifact_location: "mlflow-artifacts:/1",
      lifecycle_stage: "active",
      creation_time: epoch,
      last_update_time: epoch,
      tags: [],
    },
  };
};

export const createDecisionPopulation = (
  runCount = 8,
  { sessionCount = 3 } = {},
) => {
  const population = createPopulation(runCount);
  const examples = population.traces.slice(0, sessionCount);
  const traceDetails = {};
  const histories = examples.flatMap((example, sessionIndex) =>
    [1, 2, 3].map((step) => {
      const traceId = `tr-example-history-${sessionIndex + 1}-${step}`;
      const requestTime = new Date(
        epoch + [3000, 1000, 1000][step - 1],
      ).toISOString();
      const trace = {
        ...example,
        trace_id: traceId,
        request_time: requestTime,
        tags: {
          ...example.tags,
          step: String(step),
          "mlflow.traceName": `Decision ${step}`,
          "mlflow.trace.spansLocation": "TRACKING_STORE",
        },
        assessments:
          step === 1
            ? example.assessments.map((assessment) => ({
                ...assessment,
                trace_id: traceId,
              }))
            : [],
      };
      const session = trace.trace_metadata["mlflow.trace.session"];
      const attributes = {
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {
          source: {
            observation: `Input ${session} decision ${step}`,
            evidence: Array.from(
              { length: 20 },
              (_, index) => `Observed example fact ${index + 1}`,
            ),
          },
          details: `Additional input ${step}`,
        },
        "mlflow.spanOutputs": {
          source: {
            decision: `Output ${session} decision ${step}`,
            marker: `visible-${sessionIndex + 1}-${step}`,
          },
          details: `Additional output ${step}`,
        },
      };
      const start = BigInt(Date.parse(requestTime)) * 1_000_000n;
      traceDetails[traceId] = {
        trace_info: trace,
        spans: [
          {
            trace_id: traceId,
            span_id: (sessionIndex * 3 + step).toString(16).padStart(16, "0"),
            parent_span_id: null,
            name: `Decision ${step}`,
            kind: "SPAN_KIND_INTERNAL",
            start_time_unix_nano: String(start),
            end_time_unix_nano: String(start + 100_000_000n),
            status: { code: "STATUS_CODE_OK" },
            attributes: Object.entries(attributes).map(([key, value]) => ({
              key,
              value: { string_value: JSON.stringify(value) },
            })),
          },
        ],
      };
      return trace;
    }),
  );
  return {
    ...population,
    traces: [...histories, ...population.traces.slice(sessionCount)],
    traceDetails,
    decisionSessions: examples.map(
      (example) => example.trace_metadata["mlflow.trace.session"],
    ),
  };
};

const urgencyPreview =
  "[bold]*AGENT SYSTEM PROMPT*[/bold]\n  exact **markers**\nSystem: Estimate urgency from the recorded triage state.\nUser: Patient A has severe bleeding. Candidates: treat A, inspect B.";
const capturedUrgencyContext = `${urgencyPreview}\nCaptured note: José reports “presión” at the injury site.\nPrevious tool result: Apply direct pressure; reassess bleeding.\nOnly these recorded messages were sent to this model call.`;
const exampleLogPrefix = Array.from(
  { length: 9 },
  (_, index) => `Earlier log entry ${index + 1}: café.\n`,
).join("");
const reviewerInput =
  "System: Review the proposed action against the target.\nProposal: Apply a tourniquet to Patient A.";
const reviewerOutput =
  "Recommend assessing Patient B. Confidence: 0.8; estimated improvement: 0.4.";
const recordedProposal = {
  action: "Apply a tourniquet to Patient A.",
  justification:
    "Control Patient A’s observed severe bleeding before reassessing. The recorded observation has no assessment result for Patient B.",
  observations: Array.from(
    { length: 30 },
    (_, index) =>
      `Recorded observation ${index + 1}: Patient A has visible severe bleeding; treatment supplies are available.`,
  ),
  conclusion:
    "Captured proposal tail: control the observed bleeding before reassessment.",
};
const recordedProposalTool = {
  tool_name: "propose_treatment",
  arguments: { patient_id: "patient-a", treatment: "tourniquet" },
  note: "Proposal tool metadata remains with the proposal stage.",
};
const remainingLogLines = Array.from({ length: 20 }, () => "");
remainingLogLines[0] = '{"treat A": 0.91, "inspect B": 0.44}';
remainingLogLines[2] = '{"treat A": 0.62, "inspect B": 0.58}';
remainingLogLines[13] = reviewerInput;
remainingLogLines[18] = reviewerOutput;
const exampleLog = Buffer.from(
  `${exampleLogPrefix}${capturedUrgencyContext}\n${remainingLogLines.join("\n")}\n`,
  "utf8",
);
const exampleSnapshot = createHash("sha256").update(exampleLog).digest("hex");
const exampleArtifactPath = `evidence/log/${exampleSnapshot}/raw_align_system.log`;
const contextByteRange = [
  Buffer.byteLength(exampleLogPrefix, "utf8"),
  Buffer.byteLength(exampleLogPrefix + capturedUrgencyContext, "utf8"),
];
const logSource = (first, last) => ({
  file: "raw_align_system.log",
  lines: [first, last],
  snapshot_sha256: exampleSnapshot,
  artifact_path: exampleArtifactPath,
});
const recordSource = (step, pointer) => ({
  file: "input_output.json",
  record_index: step - 1,
  json_pointer: pointer,
});
const processPayload = (preview, source, truncated = false) => ({
  preview,
  truncated,
  source,
});

const savedActionPhase = (
  step,
  action = "Apply a tourniquet to Patient A.",
) => ({
  id: "saved-action",
  label: "Saved environment action",
  activities: [
    {
      id: "saved-action",
      kind: "action",
      status: "recorded",
      label: "Saved environment action",
      outputs: processPayload(action, recordSource(step, "/output/action")),
    },
  ],
});

const pipelineProcess = (step) => ({
  schema_version: "decision-process-v1",
  producer: "pipeline",
  phases: [
    {
      id: `tracking-${step}`,
      label: "State tracking",
      summary: "Updated the recorded patient observation.",
      activities: [
        {
          id: `observation-${step}`,
          kind: "context",
          label: "Update observed patient state",
          outputs: processPayload(
            "Patient A: severe bleeding; conscious; tourniquet available.",
            recordSource(step, "/input/state"),
          ),
        },
      ],
    },
    {
      id: `prediction-${step}`,
      label: "Predict urgency",
      summary: "Scored the visible treatment candidates.",
      activities: [
        {
          id: `urgency-${step}`,
          kind: "model",
          label: "Predict patient urgency",
          status: "complete",
          inputs: processPayload(
            urgencyPreview,
            { ...logSource(10, 16), byte_range: contextByteRange },
            true,
          ),
          outputs: processPayload(
            '{"treat A": 0.91, "inspect B": 0.44}',
            logSource(17, 17),
          ),
          source: logSource(10, 18),
        },
        {
          id: `cache-${step}`,
          kind: "cache",
          label: "Reuse candidate merit scores",
          status: "hit",
          outputs: processPayload(
            '{"treat A": 0.62, "inspect B": 0.58}',
            logSource(19, 19),
          ),
        },
      ],
    },
    savedActionPhase(step),
  ],
  gaps: ["Cached merit scores do not include a fresh model request."],
});

const reviewerProcess = (step) => ({
  schema_version: "decision-process-v1",
  producer: "reviewer",
  phases: [
    {
      id: "proposal",
      label: "Proposal",
      activities: [
        {
          id: `propose-${step}`,
          kind: "recorded",
          label: "Recorded proposal",
          outputs: processPayload(
            JSON.stringify(recordedProposal, null, 2).slice(0, 2048),
            recordSource(step, "/choice_info/multi_agent/proposal"),
            true,
          ),
        },
      ],
    },
    {
      id: "review",
      label: "Review",
      summary: "Reviewer suggested checking Patient B before treatment.",
      activities: [
        {
          id: `review-call-${step}`,
          kind: "review",
          label: "Review proposed treatment",
          inputs: processPayload(reviewerInput, logSource(30, 31)),
          outputs: processPayload(reviewerOutput, logSource(36, 36)),
          source: logSource(30, 36),
        },
      ],
    },
    {
      id: "final",
      label: "Final selection",
      activities: [
        {
          id: `final-action-${step}`,
          kind: "action",
          label: "Record executed treatment",
          outputs: processPayload(
            "Apply a tourniquet to Patient A.",
            recordSource(step, "/choice_info/multi_agent/final_action"),
          ),
        },
      ],
    },
    ...(reviewedLogSnapshots[step - 1]
      ? [
          {
            id: "unassigned",
            label: "Other recorded log sections",
            activities: reviewedLogSnapshots[step - 1].activities,
          },
        ]
      : []),
    savedActionPhase(step),
  ],
  review: {
    proposal: "Apply a tourniquet to Patient A.",
    reviewer_recommendation: "Assess Patient B before treatment.",
    computed_suggestion: "Switch to assessing Patient B.",
    final_action: "Apply a tourniquet to Patient A.",
    followed_guidance: false,
    review_mode: "advice",
    influence: 0.5,
    confidence: 0.8,
    estimated_improvement: 0.4,
    swap_threshold: 0.1,
    strength: 0.16,
  },
  gaps: [],
});

const langGraphProcess = (step) => ({
  schema_version: "decision-process-v1",
  producer: "langgraph",
  phases: [
    {
      id: `assessment-${step}`,
      label: "Assessment",
      activities: [
        {
          id: `inspect-${step}`,
          kind: "tool",
          label: "Inspect patient",
          inputs: processPayload(
            '{"patient_id":"patient-a"}',
            recordSource(
              step,
              "/choice_info/agent/proposal_history/0/tool_calls/0/args",
            ),
          ),
          outputs: processPayload(
            '{"patient_id":"patient-a","bleeding":"severe"}',
            recordSource(
              step,
              "/choice_info/agent/proposal_history/0/tool_results/0",
            ),
            true,
          ),
        },
        {
          id: `urgency-${step}`,
          kind: "model",
          label: "Classify observed urgency",
          outputs: processPayload(
            "Immediate treatment is indicated by the observed bleeding.",
            recordSource(
              step,
              "/choice_info/agent/proposal_history/0/response",
            ),
          ),
        },
      ],
    },
    {
      id: `treatment-${step}`,
      label: "Treatment",
      activities: [
        {
          id: `treatment-call-${step}`,
          kind: "tool",
          label: "Choose available treatment",
          inputs: processPayload(
            '{"patient_id":"patient-a","treatment":"tourniquet"}',
            recordSource(
              step,
              "/choice_info/agent/proposal_history/1/tool_calls/0/args",
            ),
          ),
          outputs: processPayload(
            "Apply a tourniquet to Patient A.",
            recordSource(
              step,
              "/choice_info/agent/proposal_history/1/tool_results/0",
            ),
          ),
        },
      ],
    },
    savedActionPhase(step),
  ],
  gaps: ["The model request before urgency classification was not captured."],
});

const logSnapshotFixture = (id, excerpts) => {
  const log = Buffer.from(
    `${excerpts.map(({ text }) => text).join("\n")}\n`,
    "utf8",
  );
  const snapshot = createHash("sha256").update(log).digest("hex");
  const artifactPath = `evidence/log/${snapshot}/raw_align_system.log`;
  let firstLine = 1;
  const activities = excerpts.map((excerpt, index) => {
    const lastLine = firstLine + excerpt.text.split("\n").length - 1;
    const source = {
      file: "raw_align_system.log",
      lines: [firstLine, lastLine],
      snapshot_sha256: snapshot,
      artifact_path: artifactPath,
    };
    firstLine = lastLine + 1;
    return {
      id: `${id}-${index}`,
      kind: "recorded",
      label: excerpt.label,
      [excerpt.label === "AGENT SYSTEM PROMPT" ? "inputs" : "outputs"]:
        processPayload(excerpt.text, source),
    };
  });
  return { artifactPath, log, activities };
};

const reviewedLogSnapshots = [1, 2].map((step) =>
  logSnapshotFixture(`reviewed-log-${step}`, [
    {
      label: "AGENT",
      text: "[bold]*AGENT*[/bold]: I will propose treatment for Patient A.",
    },
    {
      label: "AGENT TOOL CALL",
      text:
        `[bold]*AGENT TOOL CALL*[/bold]: ${recordedProposalTool.tool_name}(${JSON.stringify(recordedProposalTool.arguments)})` +
        (step === 1
          ? '\nAction review: {\n  "recommendation": "Assess Patient B before treatment."\n}'
          : ""),
    },
    {
      label: "AGENT",
      text: "[bold]*AGENT*[/bold]: I will keep the proposal after considering the recommendation.",
    },
    {
      label: "AGENT TOOL CALL",
      text: '[bold]*AGENT TOOL CALL*[/bold]: list_treatments({"patient_id":"patient-b"})',
    },
  ]),
);

const legacyLogSnapshots = [1, 2].map((step) => {
  const excerpts = [
    ...(step === 1
      ? [
          {
            label: "AGENT SYSTEM PROMPT",
            text: "[bold]*AGENT SYSTEM PROMPT*[/bold]:\nUse the recorded observations to select an available treatment.",
          },
        ]
      : []),
    {
      label: "AGENT THINKING",
      text: "[bold]*AGENT THINKING*[/bold]:\nCheck the observed bleeding against the available treatment options.",
    },
    {
      label: "AGENT",
      text: "[bold]*AGENT*[/bold]:\nPatient A has severe bleeding; inspect the recorded treatment options.",
    },
    {
      label: "AGENT TOOL CALL",
      text: "[bold]*AGENT TOOL CALL*[/bold]: list_available_actions({})",
    },
    {
      label: "AGENT LISTING AVAILABLE ACTIONS",
      text: "[bold]*AGENT LISTING AVAILABLE ACTIONS*[/bold]\n- inspect_patient: Inspect a selected patient.\n- list_treatments: List available treatments.",
    },
    {
      label: "AGENT TOOL CALL",
      text: "[bold]*AGENT TOOL CALL*[/bold]: observe_environment({})",
    },
    {
      label: "AGENT OBSERVING ENVIRONMENT",
      text: "[bold]*AGENT OBSERVING ENVIRONMENT*[/bold]",
    },
    {
      label: "AGENT TOOL CALL",
      text: '[bold]*AGENT TOOL CALL*[/bold]: list_treatments({"patient_id":"patient-b","review_response":"Keep the recorded treatment choice."})',
    },
    {
      label: "AGENT TOOL CALL",
      text: '[bold]*AGENT TOOL CALL*[/bold]: inspect_patient({"patient_id":oops})',
    },
  ];
  return logSnapshotFixture(`unassociated-log-${step}`, excerpts);
});

const legacyProcess = (step) => ({
  schema_version: "decision-process-v1",
  producer: "unknown",
  phases: [
    {
      id: "unassigned",
      label: "Other recorded log sections",
      summary:
        "Component ownership is unknown; these sections are not attributed to individual calls.",
      activities: legacyLogSnapshots[step - 1]?.activities ?? [],
    },
    savedActionPhase(step, "Treat Patient A for severe bleeding."),
  ].filter((phase) => phase.activities.length > 0),
  gaps: [
    "No intermediate ADM activity can be associated with the final decision record.",
    "The unassociated log excerpt cannot establish which model call produced this action.",
    "Some log sections have no verified component owner and are shown without call attribution.",
  ],
});

export const createProcessDecisionPopulation = (runCount = 8) => {
  const population = createDecisionPopulation(runCount, { sessionCount: 4 });
  const producers = [
    pipelineProcess,
    reviewerProcess,
    langGraphProcess,
    legacyProcess,
  ];
  for (const [sessionIndex, session] of population.decisionSessions.entries()) {
    for (let step = 1; step <= 3; step++) {
      const traceId = `tr-example-history-${sessionIndex + 1}-${step}`;
      const root = population.traceDetails[traceId].spans[0];
      const process = producers[sessionIndex](step);
      const rawState = {
        scenario_id: "example-triage-scenario",
        characters: [
          { id: "patient-a", visible: true, bleeding: "severe" },
          { id: "patient-b", visible: true, bleeding: "unknown" },
        ],
        description: `Recorded environment for ${session} decision ${step}`,
      };
      const rawAction = {
        action_type: "APPLY_TREATMENT",
        character_id: "patient-a",
        unstructured:
          sessionIndex === 3
            ? "Treat Patient A for severe bleeding."
            : "Apply a tourniquet to Patient A.",
      };
      const choiceInfo =
        sessionIndex === 1
          ? {
              multi_agent: {
                proposal: recordedProposal,
                proposal_tool: recordedProposalTool,
                proposal_validation: null,
                advice: {
                  recommendation: "Assess Patient B before treatment.",
                  reason:
                    "Assess Patient B to check whether an unobserved injury also needs immediate care.",
                },
                review_assessment_validation: {
                  valid: true,
                  note: "Reviewer assessment validation remains with review.",
                },
                review_status: "reviewed",
                review_response_contract: "advice",
                weighting: {
                  configuration: { influence: 0.5, swap_threshold: 0.1 },
                  confidence: 0.8,
                  estimated_improvement: 0.4,
                  strength: 0.16,
                  suggestion: "Switch to assessing Patient B.",
                  followed_guidance: false,
                },
                final_action: rawAction,
                final_review_response: {
                  response_to_advice:
                    "I kept the proposal because Patient A’s bleeding was observed and requires treatment now.",
                },
                disposition: "kept_proposal",
                calls: { proposal: 1, review: 1, final: 1 },
                unclassified_note:
                  "Unknown multi-agent metadata remains available once.",
              },
            }
          : sessionIndex === 2
            ? {
                agent: {
                  proposal_history: [
                    {
                      tool_calls: [
                        {
                          name: "inspect_patient",
                          args: { patient_id: "patient-a" },
                        },
                      ],
                      tool_results: [
                        {
                          patient_id: "patient-a",
                          bleeding: "severe",
                          assessment_detail:
                            "Complete recorded observation: capillary refill is delayed.",
                        },
                      ],
                      response:
                        "Immediate treatment is indicated by the observed bleeding.",
                    },
                    {
                      tool_calls: [
                        {
                          name: "choose_treatment",
                          args: {
                            patient_id: "patient-a",
                            treatment: "tourniquet",
                          },
                        },
                      ],
                      tool_results: [rawAction],
                    },
                  ],
                },
              }
            : {};
      const detailedSummary = sessionIndex === 0 && step === 1;
      const situationSummary = `**Situation**\n\nPatient A has severe bleeding. Patient B is awaiting assessment.\n\nDecision ${step} environment summary.`;
      const actionSummary =
        "**Executed action:** Apply a tourniquet to Patient A.\n\n**Justification:** Control the observed severe bleeding.";
      const attributes = {
        "mlflow.spanType": "UNKNOWN",
        "align.card_format": "situation-action-v1",
        "mlflow.spanInputs": {
          input: detailedSummary
            ? situationSummary +
              "\n\nThe available decision set includes treating Patient A and assessing Patient B. No assessment result for Patient B is part of this observation. Patient A is the subject of the recorded treatment proposal. The available supplies include a tourniquet. These details describe the same visible observation and available choices throughout this decision; they do not add a treatment outcome or findings from an unperformed assessment.\n\nEnd of situation summary: Patient B still awaits assessment."
            : situationSummary,
          source: { state: rawState },
        },
        "mlflow.spanOutputs": {
          response: detailedSummary
            ? actionSummary +
              "\n\nThe proposal uses the treatment supplies already listed in the observation. Its stated justification addresses the severe bleeding described for Patient A. Assessment of Patient B remains an available alternative in the same decision set. This summary describes the selected action and its recorded justification without asserting additional examination findings or a completed treatment result.\n\nEnd of action summary: no later treatment outcome is asserted."
            : actionSummary,
          evidence: { record_index: step - 1, process },
          source: {
            output: { action: rawAction },
            choice_info: choiceInfo,
            recorded_debug_note:
              "Metadata remains associated with the decision.",
          },
        },
      };
      root.attributes = Object.entries(attributes).map(([key, value]) => ({
        key,
        value: { string_value: JSON.stringify(value) },
      }));
    }
  }
  return {
    ...population,
    sourceArtifacts: {
      [exampleArtifactPath]: exampleLog,
      ...Object.fromEntries(
        [...legacyLogSnapshots, ...reviewedLogSnapshots].map(
          ({ artifactPath, log }) => [artifactPath, log],
        ),
      ),
    },
  };
};

const filterRows = (rows, filter, readField) => {
  if (!filter?.trim()) return rows;
  const token =
    /\s*((?:tags|request_metadata|attributes)\.(?:`[^`]+`|"[^"]+"|[\w.]+)|'(?:\\.|''|[^'\\])*'|AND\b|IN\b|[=(),])/giy;
  const tokens = [];
  let offset = 0;
  while (offset < filter.trimEnd().length) {
    token.lastIndex = offset;
    const match = token.exec(filter);
    if (!match) throw new Error("Unsupported fixture filter: " + filter);
    tokens.push(match[1]);
    offset = token.lastIndex;
  }
  let position = 0;
  const take = () => tokens[position++];
  const expectToken = (expected) => {
    if (take()?.toUpperCase() !== expected)
      throw new Error("Expected " + expected + " in fixture filter: " + filter);
  };
  const literal = () => {
    const value = take();
    if (!value?.startsWith("'"))
      throw new Error("Expected quoted fixture value: " + filter);
    return value
      .slice(1, -1)
      .replace(/''/g, "'")
      .replace(/\\'/g, "'")
      .replace(/\\\\/g, "\\");
  };
  const clauses = [];
  while (position < tokens.length) {
    const field = take();
    if (!/^(tags|request_metadata|attributes)\./i.test(field || ""))
      throw new Error("Expected fixture field: " + filter);
    const operator = take()?.toUpperCase();
    const values = [];
    if (operator === "=") values.push(literal());
    else if (operator === "IN") {
      expectToken("(");
      values.push(literal());
      while (tokens[position] === ",") {
        take();
        values.push(literal());
      }
      expectToken(")");
    } else throw new Error("Unsupported fixture operator: " + operator);
    const dot = field.indexOf(".");
    const namespace = field.slice(0, dot).toLowerCase();
    const key = field.slice(dot + 1).replace(/^[`"]|[`"]$/g, "");
    clauses.push((row) => values.includes(readField(row, namespace, key)));
    if (position < tokens.length) expectToken("AND");
  }
  return rows.filter((row) => clauses.every((clause) => clause(row)));
};

const traceField = (trace, namespace, key) => {
  if (namespace === "tags") return trace.tags[key];
  if (namespace === "request_metadata") return trace.trace_metadata[key];
  const attributes = {
    run_id: trace.trace_metadata["mlflow.sourceRun"],
    status: trace.state,
    name: trace.tags["mlflow.traceName"],
  };
  if (!(key in attributes))
    throw new Error("Unsupported trace attribute: " + key);
  return attributes[key];
};

const runField = (run, namespace, key) => {
  if (namespace === "tags")
    return Object.fromEntries(run.data.tags.map((tag) => [tag.key, tag.value]))[
      key
    ];
  if (namespace !== "attributes" || !(key in run.info))
    throw new Error("Unsupported run field: " + namespace + "." + key);
  return String(run.info[key]);
};

export const installApiFixtures = async (page, population) => {
  const unexpected = [];
  await page.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== "http://127.0.0.1:4173") {
      await route.abort();
      return;
    }
    const name = url.pathname;
    if (name === "/version") {
      await route.fulfill({ contentType: "text/plain", body: "3.17.2" });
      return;
    }
    if (name.endsWith("/get-artifact") && population.sourceArtifacts) {
      const artifact = population.sourceArtifacts[url.searchParams.get("path")];
      const runId = url.searchParams.get("run_uuid");
      const range = /^bytes=(\d+)-(\d+)$/.exec(
        request.headers()["range"] || "",
      );
      const start = Number(range?.[1]);
      const end = Number(range?.[2]);
      if (
        !artifact ||
        !population.runs.some((run) => run.info.run_id === runId) ||
        !range ||
        start < 0 ||
        end < start ||
        end >= artifact.length
      ) {
        unexpected.push(
          `${request.method()} ${name} with invalid source range`,
        );
        await route.fulfill({ status: 416, body: "Invalid source range" });
        return;
      }
      await route.fulfill({
        status: 206,
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Content-Range": `bytes ${start}-${end}/${artifact.length}`,
          "Accept-Ranges": "bytes",
        },
        body: artifact.subarray(start, end + 1),
      });
      return;
    }
    if (!/^\/(?:ajax-api|api|graphql)(?:\/|$)/.test(name)) {
      await route.continue();
      return;
    }
    const body =
      request.method() === "POST"
        ? request.postDataJSON() || {}
        : Object.fromEntries(url.searchParams);
    let json;
    if (name.endsWith("/users/current")) {
      await route.fulfill({
        status: 404,
        json: { message: "Authentication is disabled" },
      });
      return;
    }
    if (name.endsWith("/server-info"))
      json = {
        store_type: "SqlAlchemyStore",
        workspaces_enabled: false,
        trace_archival_enabled: false,
        multipart_uploads_enabled: false,
        multipart_downloads_enabled: false,
        features_enabled: { gateway: false },
      };
    else if (name.endsWith("/assistant/config"))
      json = { providers: {}, projects: {}, remote_access_allowed: false };
    else if (name.endsWith("/assistant/providers"))
      json = { providers: [], resolved: null };
    else if (
      name === "/graphql" &&
      body.operationName === "MlflowGetExperimentQuery"
    ) {
      if (body.variables?.input?.experimentId !== "1")
        throw new Error("Unexpected GraphQL experiment");
      json = {
        data: {
          mlflowGetExperiment: {
            __typename: "MlflowGetExperimentResponse",
            apiError: null,
            experiment: {
              __typename: "MlflowExperiment",
              artifactLocation: population.experiment.artifact_location,
              creationTime: String(epoch),
              effectiveTraceArchivalRetention: null,
              experimentId: "1",
              lastUpdateTime: String(epoch),
              lifecycleStage: "active",
              name: population.experiment.name,
              tags: [],
            },
          },
        },
      };
    } else if (name.endsWith("/experiments/get"))
      json = { experiment: population.experiment };
    else if (name.endsWith("/experiments/search"))
      json = { experiments: [population.experiment] };
    else if (name.endsWith("/traces/search")) {
      const offset = Number(body.page_token || 0);
      const limit = Number(body.max_results || 500);
      const matches = filterRows(
        population.traces,
        body.filter || body.filter_string,
        traceField,
      );
      const traces = matches.slice(offset, offset + limit);
      json = {
        traces,
        ...(offset + limit < matches.length
          ? { next_page_token: String(offset + limit) }
          : {}),
      };
    } else if (name.endsWith("/traces/get")) {
      const trace = population.traceDetails?.[body.trace_id];
      if (!trace)
        throw new Error("No detailed trace fixture for " + body.trace_id);
      json = { trace };
    } else if (/\/3\.0\/mlflow\/traces\/[^/]+$/.test(name)) {
      const trace =
        population.traceDetails?.[decodeURIComponent(name.split("/").at(-1))];
      if (!trace) throw new Error("No trace info fixture for " + name);
      json = { trace: { trace_info: trace.trace_info } };
    } else if (name.endsWith("/get-trace-artifact")) {
      const trace = population.traceDetails?.[body.request_id];
      if (!trace)
        throw new Error("No trace artifact fixture for " + body.request_id);
      json = { spans: trace.spans };
    } else if (name.endsWith("/datasets/search")) {
      json = { datasets: [] };
    } else if (name.endsWith("/runs/search")) {
      const offset = Number(body.page_token || 0);
      const limit = Number(body.max_results || 50);
      const matches = filterRows(
        population.runs,
        body.filter || body.filter_string,
        runField,
      );
      json = {
        runs: matches.slice(offset, offset + limit),
        ...(offset + limit < matches.length
          ? { next_page_token: String(offset + limit) }
          : {}),
      };
    } else if (name.endsWith("/runs/get"))
      json = {
        run: population.runs.find((run) => run.info.run_uuid === body.run_id),
      };
    else {
      unexpected.push(`${request.method()} ${name}`);
      await route.fulfill({
        status: 404,
        json: {
          error_code: "RESOURCE_DOES_NOT_EXIST",
          message: "No API fixture for this request",
        },
      });
      return;
    }
    await route.fulfill({ json });
  });
  return unexpected;
};
