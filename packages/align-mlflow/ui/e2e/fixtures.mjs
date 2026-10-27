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
