import { test, expect } from "./browser.mjs";

test.use({ processEvidence: true, viewport: { width: 960, height: 1000 } });

const sessionRoute = (sessions) => {
  const query = new URLSearchParams();
  for (const session of sessions.slice(1))
    query.append("compareSessionId", session);
  return (
    "/#/experiments/1/chat-sessions/" +
    encodeURIComponent(sessions[0]) +
    (query.size ? "?" + query : "")
  );
};
const card = (scope, step = 1) =>
  scope.getByRole("group", { name: `Decision ${step}`, exact: true });
const summary = (scope, name) =>
  scope.locator("summary").filter({
    hasText: new RegExp(
      "^" + name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?:\\s*·|$)",
    ),
  });
const navigation = (page, session) =>
  page.getByRole("navigation", { name: `${session} decisions`, exact: true });

const closeAssistant = async (page) => {
  const close = page
    .locator('[data-assistant-ui="true"]')
    .getByRole("button", { name: "Close", exact: true });
  await close.click();
  await expect(close).not.toBeVisible();
};

const expectSelection = async (page, scope, session, selected) => {
  for (let step = 1; step <= 3; step++) {
    const value = String(step === selected);
    await expect(
      navigation(page, session).getByRole("button", {
        name: `Decision ${step}`,
        exact: true,
      }),
    ).toHaveAttribute("aria-pressed", value);
    await expect(
      card(scope, step).getByRole("button", {
        name: `Decision ${step}`,
        exact: true,
      }),
    ).toHaveAttribute("aria-pressed", value);
  }
};

const expectRawPreview = async (scope, title) => {
  const recorded = scope.getByRole("region", { name: title, exact: true });
  await expect(
    recorded.getByRole("heading", { name: title, exact: true }),
  ).toBeVisible();
  await expect(recorded.locator("pre")).toBeVisible();
  await expect(
    recorded.locator("pre .react-syntax-highlighter-line-number"),
  ).toHaveCount(0);
  expect(
    (await recorded.locator("pre code").textContent()).trimEnd().split("\n"),
  ).toHaveLength(2);
  await expect(
    recorded.getByRole("button", { name: "See more", exact: true }),
  ).toHaveCount(1);
  await expect(
    recorded.getByRole("button", { name: "See more", exact: true }),
  ).toBeVisible();
  await expect(
    recorded.getByRole("button", { name: "See less", exact: true }),
  ).toHaveCount(0);
  return recorded;
};

const expectNoLineNumbers = async (scope) => {
  await expect(
    scope.locator(".react-syntax-highlighter-line-number"),
  ).toHaveCount(0);
  await expect(scope.getByTestId("json-line-number")).toHaveCount(0);
};

const collapseRawPreview = async (recorded) => {
  const collapse = recorded.getByRole("button", {
    name: "See less",
    exact: true,
  });
  await expect(collapse).toHaveCount(2);
  const [toggleBounds, contentBounds] = await Promise.all([
    collapse.first().boundingBox(),
    recorded.locator("pre").boundingBox(),
  ]);
  expect(toggleBounds.y).toBeLessThanOrEqual(contentBounds.y + 1);
  await collapse.first().click();
};

const expectCompactDecision = async (scope, session, phases) => {
  const decision = card(scope);
  await expect(
    decision.getByRole("heading", {
      name: /^(Environment|Selected action|Chosen action)$/,
    }),
  ).toHaveCount(0);
  await expect(decision.getByText("Situation", { exact: true })).toHaveCount(1);
  await expect(decision.getByText("Situation", { exact: true })).toBeVisible();
  const situation = decision.getByRole("region", {
    name: "Situation input",
    exact: true,
  });
  const action = decision.getByRole("region", {
    name: "Chosen action",
    exact: true,
  });
  await expect(
    situation.getByText("Patient A has severe bleeding.", { exact: false }),
  ).toBeVisible();
  await expect(
    action.getByText("Apply a tourniquet to Patient A.", { exact: false }),
  ).toBeVisible();
  await expect(summary(situation, "Recorded environment state")).toHaveCount(0);
  await expect(summary(action, "Recorded action")).toHaveCount(0);
  await expectRawPreview(situation, "Input");
  await expectRawPreview(action, "Output");
  await expectNoLineNumbers(scope);
  const [situationBounds, actionBounds] = await Promise.all([
    situation.boundingBox(),
    action.boundingBox(),
  ]);
  expect(situationBounds.y + situationBounds.height).toBeLessThanOrEqual(
    actionBounds.y + 1,
  );
  await expect(
    decision.getByText(`Recorded environment for ${session} decision 1`, {
      exact: false,
    }),
  ).not.toBeVisible();
  const process = summary(decision, `Decision process (${phases} phases)`);
  await expect(process).toBeVisible();
  await expect(
    decision.getByText("System: Estimate urgency", { exact: false }),
  ).not.toBeVisible();
  const visibleHeadings = await decision
    .locator("summary:visible")
    .allTextContents();
  expect(visibleHeadings).toEqual([`Decision process (${phases} phases)`]);
  return decision;
};

const openPipelineCall = async (decision) => {
  await summary(decision, "Decision process (3 phases)").click();
  await expect(summary(decision, "State tracking")).toBeVisible();
  await expect(summary(decision, "Predict urgency")).toBeVisible();
  await expect(summary(decision, "Select action")).toBeVisible();
  await expect(summary(decision, "Predict patient urgency")).not.toBeVisible();
  await summary(decision, "Predict urgency").click();
  const activity = summary(decision, "Predict patient urgency");
  await expect(activity).toBeVisible();
  await expect(
    decision.getByText("System: Estimate urgency", { exact: false }),
  ).not.toBeVisible();
  await activity.click();
  const details = activity.locator("..");
  await expect(
    details.getByText("System: Estimate urgency", { exact: false }),
  ).toBeVisible();
  await expect(
    details.getByRole("region", { name: "Outputs", exact: true }),
  ).toContainText(/treat A"?\s*:\s*0\.91/);
  await expect(
    details.getByRole("region", { name: "Outputs", exact: true }),
  ).toContainText(/inspect B"?\s*:\s*0\.44/);
  await expect(
    details.getByRole("button", { name: "See more", exact: true }),
  ).toHaveCount(0);
  await expectNoLineNumbers(details);
  await expect(
    decision.getByText("raw_align_system.log", { exact: false }).first(),
  ).toBeVisible();
  await expect(
    details.getByText("Captured note: José", { exact: false }),
  ).toBeVisible();
  await expect(
    details.getByText(
      "Preview only. The complete payload is available from its recorded source.",
      { exact: true },
    ),
  ).not.toBeVisible();
  const source = decision
    .getByRole("link", { name: "Open source artifact", exact: true })
    .first();
  await expect(source).toBeVisible();
  await expect(source).toHaveAttribute(
    "href",
    /evidence(?:%2F|\/)log(?:%2F|\/)[a-f0-9]{64}(?:%2F|\/)raw_align_system\.log/,
  );
  await activity.click();
  await expect(
    decision.getByText("System: Estimate urgency", { exact: false }),
  ).not.toBeVisible();
  await summary(decision, "Reuse candidate merit scores").click();
  await expect(
    decision.getByText("Input context was not captured.", { exact: true }),
  ).toHaveCount(0);
  await expect(
    decision.getByText(
      "Cached merit scores do not include a fresh model request.",
      {
        exact: true,
      },
    ),
  ).toBeVisible();
};

test("pipeline evidence expands by phase and call while the environment remains primary", async ({
  page,
  population,
}) => {
  const session = population.decisionSessions[0];
  await page.goto(sessionRoute([session]));
  await closeAssistant(page);
  const decision = await expectCompactDecision(page, session, 3);
  await navigation(page, session)
    .getByRole("button", { name: "Decision 2", exact: true })
    .click();
  await openPipelineCall(decision);
  await expectSelection(page, page, session, 2);
  const situation = decision.getByRole("region", {
    name: "Situation input",
    exact: true,
  });
  const recordedInput = await expectRawPreview(situation, "Input");
  await expect(recordedInput).not.toContainText("example-triage-scenario");
  await recordedInput
    .getByRole("button", { name: "See more", exact: true })
    .click();
  await expect(recordedInput).toContainText("example-triage-scenario");
  await expect(recordedInput).not.toContainText("APPLY_TREATMENT");
  await expect(recordedInput).not.toContainText("choice_info");
  await expect(recordedInput).not.toContainText("recorded_debug_note");
  await expectSelection(page, page, session, 2);
  await expectNoLineNumbers(recordedInput);
  await collapseRawPreview(recordedInput);
  await expectRawPreview(situation, "Input");
  await expect(recordedInput).not.toContainText("example-triage-scenario");
  await expectSelection(page, page, session, 2);
  const action = decision.getByRole("region", {
    name: "Chosen action",
    exact: true,
  });
  const recordedAction = await expectRawPreview(action, "Output");
  await expect(recordedAction).not.toContainText("APPLY_TREATMENT");
  await recordedAction
    .getByRole("button", { name: "See more", exact: true })
    .click();
  await expect(recordedAction).toContainText("APPLY_TREATMENT");
  await expect(recordedAction).not.toContainText("example-triage-scenario");
  await expect(recordedAction).not.toContainText("choice_info");
  await expect(recordedAction).not.toContainText("recorded_debug_note");
  await expectSelection(page, page, session, 2);
  await expectNoLineNumbers(recordedAction);
  await collapseRawPreview(recordedAction);
  await expectRawPreview(action, "Output");
  await expect(recordedAction).not.toContainText("APPLY_TREATMENT");
  await expectSelection(page, page, session, 2);
  const process = summary(decision, "Decision process (3 phases)").locator(
    "..",
  );
  await summary(process, "Recorded decision metadata").click();
  const recordedMetadata = summary(
    process,
    "Recorded decision metadata",
  ).locator("..");
  await expect(recordedMetadata).toContainText("choice_info");
  await expect(recordedMetadata).toContainText(
    "Metadata remains associated with the decision.",
  );
  await expect(recordedMetadata).not.toContainText("APPLY_TREATMENT");
  await expect(recordedMetadata).not.toContainText("example-triage-scenario");
  await expectNoLineNumbers(recordedMetadata);
  await expectSelection(page, page, session, 2);
  await navigation(page, session)
    .getByRole("button", { name: "Open Predict urgency phase", exact: true })
    .click();
  await expect(summary(card(page, 2), "Predict patient urgency")).toBeVisible();
  await expectSelection(page, page, session, 2);
  await expectNoLineNumbers(page);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});

test("opening a call loads its complete captured context using the recorded UTF-8 byte range", async ({
  page,
  population,
}) => {
  const sourceRequests = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.endsWith("/get-artifact"))
      sourceRequests.push(request);
  });
  const session = population.decisionSessions[0];
  await page.goto(sessionRoute([session]));
  await closeAssistant(page);
  const decision = await expectCompactDecision(page, session, 3);
  expect(sourceRequests).toHaveLength(0);
  await summary(decision, "Decision process (3 phases)").click();
  await summary(decision, "Predict urgency").click();
  const activity = summary(decision, "Predict patient urgency");
  await expect(activity).toBeVisible();
  expect(sourceRequests).toHaveLength(0);
  const responsePromise = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith("/get-artifact"),
  );
  await activity.click();
  const response = await responsePromise;
  const inputs = activity
    .locator("..")
    .getByRole("region", { name: "Inputs", exact: true });
  const renderedLog = inputs.locator("pre");
  expect(response.status()).toBe(206);
  expect(sourceRequests).toHaveLength(1);
  const request = sourceRequests[0];
  const outputs = JSON.parse(
    population.traceDetails["tr-example-history-1-1"].spans[0].attributes.find(
      (attribute) => attribute.key === "mlflow.spanOutputs",
    ).value.string_value,
  );
  const source = outputs.evidence.process.phases[1].activities[0].inputs.source;
  const [start, end] = source.byte_range;
  expect(request.headers()["range"]).toBe(`bytes=${start}-${end - 1}`);
  expect(new URL(request.url()).searchParams.get("path")).toBe(
    source.artifact_path,
  );
  expect(response.headers()["content-range"]).toBe(
    `bytes ${start}-${end - 1}/${population.sourceArtifacts[source.artifact_path].length}`,
  );
  await expect(
    inputs.getByRole("button", { name: "See more", exact: true }),
  ).toHaveCount(0);
  await expectNoLineNumbers(inputs);
  await expect(renderedLog).toContainText("[bold]*AGENT SYSTEM PROMPT*[/bold]");
  expect(await renderedLog.textContent()).toContain("  exact **markers**");
  await expect(inputs).toContainText(
    "Captured note: José reports “presión” at the injury site.",
  );
  await expect(inputs).toContainText(
    "Previous tool result: Apply direct pressure; reassess bleeding.",
  );
  await expect(
    inputs.getByRole("button", { name: "Load full payload", exact: true }),
  ).toHaveCount(0);
  await expect(
    inputs.getByText("Preview only.", { exact: false }),
  ).not.toBeVisible();
  const topCollapse = inputs
    .getByRole("button", { name: "See less", exact: true })
    .first();
  const [toggleBounds, contentBounds] = await Promise.all([
    topCollapse.boundingBox(),
    renderedLog.boundingBox(),
  ]);
  expect(toggleBounds.y).toBeLessThanOrEqual(contentBounds.y + 1);
  await expectNoLineNumbers(page);
});

const expectReviewerDistinction = async (decision) => {
  await summary(decision, "Decision process (3 phases)").click();
  const review = decision.getByRole("region", {
    name: "Reviewer guidance and final selection",
    exact: true,
  });
  await expect(review).toBeVisible();
  const value = (label) =>
    review
      .locator("dt")
      .filter({ hasText: new RegExp(`^${label}$`) })
      .locator("xpath=following-sibling::dd[1]");
  await expect(value("Proposed action")).toHaveText(
    "Apply a tourniquet to Patient A.",
  );
  await expect(value("Reviewer recommendation")).toHaveText(
    "Assess Patient B before treatment.",
  );
  await expect(value("Computed suggestion")).toHaveText(
    "Switch to assessing Patient B.",
  );
  await expect(value("Final action")).toHaveText(
    "Apply a tourniquet to Patient A.",
  );
  await expect(value("Followed guidance")).toHaveText(/no|false/i);
  await expect(value("Strength")).toHaveText("0.16");
  await expect(value("Swap threshold")).toHaveText("0.1");
  await summary(decision, "Decision proposal").click();
  const proposal = summary(decision, "Recorded proposal");
  await proposal.click();
  const proposalDetails = proposal.locator("..");
  await expect(proposalDetails).toContainText(
    "Captured proposal tail: control the observed bleeding before reassessment.",
  );
  await expect(
    proposalDetails.getByRole("button", { name: "See more", exact: true }),
  ).toHaveCount(0);
  await expectNoLineNumbers(proposalDetails);
  await summary(decision, "Review").click();
  const call = summary(decision, "Review proposed treatment");
  await call.focus();
  await call.press("Enter");
  await expect(
    decision.getByText("System: Review the proposed action", { exact: false }),
  ).toBeVisible();
};

test("reviewer advice and computed suggestion remain distinct from the executed action", async ({
  page,
  population,
}) => {
  const session = population.decisionSessions[1];
  await page.goto(sessionRoute([session]));
  await closeAssistant(page);
  const decision = await expectCompactDecision(page, session, 3);
  await expectReviewerDistinction(decision);
  await expectNoLineNumbers(page);
});

test("graph tools expose paired arguments and results without inventing missing context", async ({
  page,
  population,
}) => {
  const session = population.decisionSessions[2];
  await page.goto(sessionRoute([session]));
  await closeAssistant(page);
  const decision = await expectCompactDecision(page, session, 2);
  await summary(decision, "Decision process (2 phases)").click();
  await summary(decision, "Assessment").click();
  const tool = summary(decision, "Inspect patient");
  await tool.click();
  const toolDetails = tool.locator("..");
  await expect(
    toolDetails.getByRole("region", { name: "Inputs", exact: true }),
  ).toContainText(/patient_id"?\s*:\s*"patient-a"/);
  await expect(
    toolDetails.getByRole("region", { name: "Outputs", exact: true }),
  ).toContainText(/bleeding"?\s*:\s*"severe"/);
  await expect(
    toolDetails.getByRole("region", { name: "Outputs", exact: true }),
  ).toContainText(
    "Complete recorded observation: capillary refill is delayed.",
  );
  await expect(
    toolDetails.getByText("Preview only.", { exact: false }),
  ).not.toBeVisible();
  await summary(decision, "Classify observed urgency").click();
  await expect(
    decision.getByText("The request for this model call was not captured.", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    decision.getByText(
      "The model request before urgency classification was not captured.",
      { exact: true },
    ),
  ).toBeVisible();
  await expectNoLineNumbers(page);
});

test("legacy decision evidence keeps incomplete and unassociated logs explicit", async ({
  page,
  population,
}) => {
  const session = population.decisionSessions[3];
  await page.goto(sessionRoute([session]));
  await closeAssistant(page);
  const decision = await expectCompactDecision(page, session, 1);
  await summary(decision, "Decision process (1 phases)").click();
  await expect(
    decision.getByText(
      "Only the final decision record was captured; intermediate ADM activity is unavailable.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(
    decision.getByText(
      "The unassociated log excerpt cannot establish which model call produced this action.",
      { exact: true },
    ),
  ).toBeVisible();
  await summary(decision, "Recorded decision").click();
  await summary(decision, "Legacy decision record").click();
  await expect(
    decision.getByText("Input context was not captured.", { exact: true }),
  ).toHaveCount(0);
  await expect(
    decision.getByText("Treat Patient A for severe bleeding.", {
      exact: false,
    }),
  ).toBeVisible();
  await expectNoLineNumbers(page);
});

test("comparison panes share process disclosures and retain independently selected decisions", async ({
  page,
  population,
}) => {
  const sessions = population.decisionSessions.slice(0, 2);
  await page.goto(sessionRoute(sessions));
  await closeAssistant(page);
  const panes = sessions.map((session) =>
    page.getByRole("region", { name: session, exact: true }),
  );
  for (let index = 0; index < sessions.length; index++) {
    await expectCompactDecision(panes[index], sessions[index], 3);
    await navigation(page, sessions[index])
      .getByRole("button", { name: "Decision 2", exact: true })
      .click();
  }
  await openPipelineCall(card(panes[0]));
  await expectReviewerDistinction(card(panes[1]));
  for (let index = 0; index < sessions.length; index++)
    await expectSelection(page, panes[index], sessions[index], 2);
  await expect(
    page.getByText("2 decisions selected", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: "Compare selected decisions",
      exact: true,
    }),
  ).toBeEnabled();
  await expectNoLineNumbers(page);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});
