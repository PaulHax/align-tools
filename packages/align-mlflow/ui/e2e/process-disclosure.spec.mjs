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
    has: scope.page().getByText(name, { exact: true }),
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
  ).toHaveCount(0);
  await expect(
    recorded.getByText(
      `input_output.json → ${title === "Input" ? "input" : "output"}`,
      { exact: true },
    ),
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

const expectStaticEvidence = async (scope) => {
  await expect(scope.locator("details")).toHaveCount(0);
  await expect(
    scope.getByRole("button", { name: /^See (more|less)$/, exact: true }),
  ).toHaveCount(0);
  await expect(
    scope.getByRole("link", { name: /^See (more|less)$/, exact: true }),
  ).toHaveCount(0);
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

const expectObjectDisplayAndCopy = async (
  page,
  recorded,
  expected,
  firstKey,
) => {
  const displayed = (
    await recorded.locator("pre code").textContent()
  ).trimEnd();
  expect(displayed.startsWith(`"${firstKey}": {\n`)).toBe(true);
  expect(JSON.parse(`{\n${displayed}\n}`)).toEqual(expected);
  await recorded
    .locator('[data-component-id="shared.model-trace-explorer.copy-snippet"]')
    .click();
  await expect
    .poll(() => page.evaluate(() => navigator.clipboard.readText()))
    .toBe(JSON.stringify(expected, null, 2));
  expect(
    JSON.parse(await page.evaluate(() => navigator.clipboard.readText())),
  ).toEqual(expected);
};

const expandAndCollapseSummary = async (
  page,
  decision,
  session,
  name,
  tail,
) => {
  const readable = decision.getByRole("region", { name, exact: true });
  const expand = readable.getByRole("button", {
    name: "See more",
    exact: true,
  });
  await expect(expand).toBeVisible();
  const preview = await readable.innerText();
  expect(preview).toContain("...");
  await expect(readable).not.toContainText(tail);
  await expand.click();
  await expect(readable).toContainText(tail);
  const collapse = readable.getByRole("button", {
    name: "See less",
    exact: true,
  });
  await expect(collapse).toHaveCount(2);
  await expect(
    readable.getByRole("link", { name: "See less", exact: true }),
  ).toHaveCount(0);
  const topCollapse = collapse.first();
  const [toggleBounds, contentBounds] = await Promise.all([
    topCollapse.boundingBox(),
    readable.getByText(tail, { exact: false }).boundingBox(),
  ]);
  expect(toggleBounds.y).toBeLessThanOrEqual(contentBounds.y + 1);
  await expectSelection(page, page, session, 2);
  await topCollapse.click();
  await expect(readable).not.toContainText(tail);
  await expect(expand).toBeVisible();
  expect(await readable.innerText()).toBe(preview);
  await expect(collapse).toHaveCount(0);
  await expectSelection(page, page, session, 2);
};

const expectCompactDecision = async (page, scope, session, stages) => {
  const decision = card(scope);
  await expect(navigation(page, session).getByRole("button")).toHaveCount(3);
  for (let step = 1; step <= 3; step++)
    await expect(
      navigation(page, session).getByRole("button", {
        name: `Decision ${step}`,
        exact: true,
      }),
    ).toBeVisible();
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
  await expect(
    decision.getByText(`Recorded environment for ${session} decision 1`, {
      exact: false,
    }),
  ).not.toBeVisible();
  const process = decision.getByRole("region", {
    name: "Decision process",
    exact: true,
  });
  await expect(process).toBeVisible();
  const processHeading = process.getByRole("heading", {
    name: "Decision process",
    level: 4,
    exact: true,
  });
  await expect(processHeading).toBeVisible();
  await expect(process.getByText(/^\d+ stages?$/, { exact: true })).toHaveCount(
    0,
  );
  await expect(
    decision.locator("summary").filter({ hasText: /^Decision process/ }),
  ).toHaveCount(0);
  const workflow = process.getByRole("region", {
    name: "Workflow stages",
    exact: true,
  });
  await expect(workflow).toHaveCount(stages.length > 0 ? 1 : 0);
  await expect(workflow.locator("summary")).toHaveCount(stages.length);
  for (const name of stages) {
    const stage = summary(workflow, name);
    await expect(stage).toBeVisible();
    const fragments = await stage.evaluate((element) => {
      const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
      const range = document.createRange();
      const boxes = [];
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        if (!node.textContent.trim()) continue;
        range.selectNodeContents(node);
        boxes.push(
          ...Array.from(range.getClientRects(), (rect) => rect.toJSON()),
        );
      }
      return boxes.filter(({ width, height }) => width > 0 && height > 0);
    });
    expect(fragments.length).toBeGreaterThan(1);
    const top = Math.min(...fragments.map(({ top }) => top));
    const bottom = Math.max(...fragments.map(({ bottom }) => bottom));
    const lineHeight = Math.max(...fragments.map(({ height }) => height));
    expect(bottom - top).toBeLessThanOrEqual(lineHeight + 2);
  }
  const supporting = decision.getByRole("region", {
    name: "Supporting evidence",
    exact: true,
  });
  await expect(supporting).toBeVisible();
  const sectionBounds = await Promise.all(
    [situation, process, action, supporting].map((region) =>
      region.boundingBox(),
    ),
  );
  for (let index = 1; index < sectionBounds.length; index++) {
    const previous = sectionBounds[index - 1];
    expect(previous.y + previous.height).toBeLessThanOrEqual(
      sectionBounds[index].y + 1,
    );
  }
  const supportingHeading = supporting.getByRole("heading", {
    name: "Supporting evidence",
    level: 4,
    exact: true,
  });
  await expect(supportingHeading).toBeVisible();
  const typography = (element) => {
    const style = getComputedStyle(element);
    return Object.fromEntries(
      ["fontFamily", "fontSize", "fontWeight", "lineHeight", "color"].map(
        (property) => [property, style[property]],
      ),
    );
  };
  const [processTypography, supportingTypography] = await Promise.all([
    processHeading.evaluate(typography),
    supportingHeading.evaluate(typography),
  ]);
  expect(processTypography).toEqual(supportingTypography);
  await expect(
    workflow.getByRole("region", {
      name: "Supporting evidence",
      exact: true,
    }),
  ).toHaveCount(0);
  for (const label of [
    "Saved environment action",
    "Other recorded log sections",
    "Capture gaps",
    "Choice info",
  ])
    await expect(summary(workflow, label)).toHaveCount(0);
  await expect(summary(decision, "Saved environment action")).toHaveCount(0);
  await expect(summary(supporting, "Additional metadata")).toHaveCount(0);
  await expect(supporting).not.toContainText("recorded_debug_note");
  await expect(
    process.getByRole("region", {
      name: "Reviewer guidance and final selection",
      exact: true,
    }),
  ).not.toBeVisible();
  await expect(
    process.getByRole("region", { name: "Capture gaps", exact: true }),
  ).not.toBeVisible();
  await expect(
    process.getByRole("region", { name: /^(Inputs|Outputs)$/, exact: true }),
  ).toHaveCount(0);
  return decision;
};

const openPipelineStage = async (decision) => {
  await expect(summary(decision, "State tracking")).toBeVisible();
  await expect(summary(decision, "Predict urgency")).toBeVisible();
  await expect(summary(decision, "Select action")).toHaveCount(0);
  await expect(summary(decision, "Predict urgency")).toContainText(
    "Scored the visible treatment candidates.",
  );
  const phase = summary(decision, "Predict urgency").locator("..");
  const recorded = phase.getByRole("region", {
    name: "Predict urgency records",
    exact: true,
  });
  await expect(recorded).toHaveCount(0);
  await summary(decision, "Predict urgency").click();
  await expect(
    recorded.getByRole("region", {
      name: "Reuse candidate merit scores",
      exact: true,
    }),
  ).toBeVisible();
  const details = recorded.getByRole("region", {
    name: "Predict patient urgency",
    exact: true,
  });
  await expect(details).toBeVisible();
  await expect(
    details.getByText("System: Estimate urgency", { exact: false }),
  ).toBeVisible();
  await expect(
    details.getByRole("region", { name: "Outputs", exact: true }),
  ).toContainText(/treat A"?\s*:\s*0\.91/);
  await expect(
    details.getByRole("region", { name: "Outputs", exact: true }),
  ).toContainText(/inspect B"?\s*:\s*0\.44/);
  await expectStaticEvidence(phase);
  await expectNoLineNumbers(details);
  await expect(
    recorded.getByText("raw_align_system.log", { exact: false }).first(),
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
  const source = recorded
    .getByRole("link", { name: "Open source artifact", exact: true })
    .first();
  await expect(source).toBeVisible();
  await expect(source).toHaveAttribute(
    "href",
    /evidence(?:%2F|\/)log(?:%2F|\/)[a-f0-9]{64}(?:%2F|\/)raw_align_system\.log/,
  );
  await expect(
    decision.getByText("Input context was not captured.", { exact: true }),
  ).toHaveCount(0);
  await summary(decision, "Predict urgency").click();
  await expect(details).toHaveCount(0);
  const captureGaps = summary(decision, "Capture gaps");
  await expect(captureGaps).toContainText(
    "Cached merit scores do not include a fresh model request.",
  );
  await captureGaps.click();
  await expect(
    decision
      .getByRole("region", { name: "Capture gaps", exact: true })
      .getByText("Cached merit scores do not include a fresh model request.", {
        exact: true,
      }),
  ).toBeVisible();
};

test("pipeline stages reveal readable context and recorded evidence while the environment remains primary", async ({
  page,
  population,
}) => {
  const session = population.decisionSessions[0];
  await page.goto(sessionRoute([session]));
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"]);
  await closeAssistant(page);
  const decision = await expectCompactDecision(page, page, session, [
    "State tracking",
    "Predict urgency",
  ]);
  await navigation(page, session)
    .getByRole("button", { name: "Decision 2", exact: true })
    .click();
  await expandAndCollapseSummary(
    page,
    decision,
    session,
    "Situation summary",
    "End of situation summary: Patient B still awaits assessment.",
  );
  await expandAndCollapseSummary(
    page,
    decision,
    session,
    "Action summary",
    "End of action summary: no later treatment outcome is asserted.",
  );
  await openPipelineStage(decision);
  await expectSelection(page, page, session, 2);
  const situation = decision.getByRole("region", {
    name: "Situation input",
    exact: true,
  });
  const root = population.traceDetails["tr-example-history-1-1"].spans[0];
  const spanValue = (key) =>
    JSON.parse(
      root.attributes.find((attribute) => attribute.key === key).value
        .string_value,
    );
  const recordedInput = await expectRawPreview(situation, "Input");
  const inputTail = `Recorded environment for ${session} decision 1`;
  await expect(recordedInput).not.toContainText(inputTail);
  await recordedInput
    .getByRole("button", { name: "See more", exact: true })
    .click();
  await expect(recordedInput).toContainText("example-triage-scenario");
  await expect(recordedInput).not.toContainText("APPLY_TREATMENT");
  await expect(recordedInput).not.toContainText("choice_info");
  await expect(recordedInput).not.toContainText("recorded_debug_note");
  await expect(recordedInput).toContainText(inputTail);
  await expectObjectDisplayAndCopy(
    page,
    recordedInput,
    spanValue("mlflow.spanInputs").source,
    "state",
  );
  await expectSelection(page, page, session, 2);
  await expectNoLineNumbers(recordedInput);
  await collapseRawPreview(recordedInput);
  await expectRawPreview(situation, "Input");
  await expect(recordedInput).not.toContainText(inputTail);
  await expectSelection(page, page, session, 2);
  const action = decision.getByRole("region", {
    name: "Chosen action",
    exact: true,
  });
  const recordedAction = await expectRawPreview(action, "Output");
  await expect(recordedAction).not.toContainText("unstructured");
  await recordedAction
    .getByRole("button", { name: "See more", exact: true })
    .click();
  await expect(recordedAction).toContainText("APPLY_TREATMENT");
  await expect(recordedAction).not.toContainText("example-triage-scenario");
  await expect(recordedAction).not.toContainText("choice_info");
  await expect(recordedAction).not.toContainText("recorded_debug_note");
  await expectObjectDisplayAndCopy(
    page,
    recordedAction,
    spanValue("mlflow.spanOutputs").source.output,
    "action",
  );
  await expectSelection(page, page, session, 2);
  await expectNoLineNumbers(recordedAction);
  await collapseRawPreview(recordedAction);
  await expectRawPreview(action, "Output");
  await expect(recordedAction).not.toContainText("unstructured");
  await expectSelection(page, page, session, 2);
  await summary(card(page, 2), "Predict urgency").click();
  const selectedPhase = summary(card(page, 2), "Predict urgency").locator("..");
  await expect(
    selectedPhase.getByRole("region", {
      name: "Predict urgency records",
      exact: true,
    }),
  ).toBeVisible();
  await expectStaticEvidence(selectedPhase);
  await expectSelection(page, page, session, 2);
  await expectNoLineNumbers(page);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});

test("opening a stage loads complete captured context using the recorded UTF-8 byte range", async ({
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
  const decision = await expectCompactDecision(page, page, session, [
    "State tracking",
    "Predict urgency",
  ]);
  expect(sourceRequests).toHaveLength(0);
  await expect(
    decision.getByRole("region", {
      name: "Predict patient urgency",
      exact: true,
    }),
  ).toHaveCount(0);
  const stage = summary(decision, "Predict urgency");
  const responsePromise = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith("/get-artifact"),
  );
  await stage.click();
  const response = await responsePromise;
  const phase = stage.locator("..");
  const recorded = phase.getByRole("region", {
    name: "Predict urgency records",
    exact: true,
  });
  const inputs = recorded
    .getByRole("region", { name: "Predict patient urgency", exact: true })
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
  await expectStaticEvidence(phase);
  await expectNoLineNumbers(page);
});

const expectReviewerDistinction = async (decision, population) => {
  const proposalJustification =
    "Control Patient A’s observed severe bleeding before reassessing. The recorded observation has no assessment result for Patient B.";
  const adviceReason =
    "Assess Patient B to check whether an unobserved injury also needs immediate care.";
  const finalResponse =
    "I kept the proposal because Patient A’s bleeding was observed and requires treatment now.";
  await expect(
    summary(decision, "Reviewer guidance and final selection"),
  ).toHaveCount(0);
  const workflow = decision.getByRole("region", {
    name: "Workflow stages",
    exact: true,
  });
  const detailValue = (scope, label) =>
    scope
      .locator("dt")
      .filter({ hasText: new RegExp(`^${label}$`) })
      .locator("xpath=following-sibling::dd[1]");
  await expect(summary(workflow, "Proposal")).toContainText(
    "Apply a tourniquet to Patient A.",
  );
  await expect(summary(workflow, "Proposal")).toContainText(
    proposalJustification,
  );
  await expect(summary(workflow, "Review")).toContainText(
    "Assess Patient B before treatment.",
  );
  await expect(summary(workflow, "Review")).toContainText(adviceReason);
  await expect(summary(workflow, "Final selection")).toContainText(
    "Apply a tourniquet to Patient A.",
  );
  await expect(summary(workflow, "Final selection")).toContainText(
    finalResponse,
  );
  await summary(workflow, "Proposal").click();
  const proposalPhase = summary(workflow, "Proposal").locator("..");
  await expect(detailValue(proposalPhase, "Proposed action")).toHaveText(
    "Apply a tourniquet to Patient A.",
  );
  const proposalProse = proposalPhase.getByText(proposalJustification, {
    exact: true,
  });
  await expect(proposalProse).toBeVisible();
  const proposalRecorded = proposalPhase.getByRole("region", {
    name: "Proposal records",
    exact: true,
  });
  await expect(proposalRecorded).toBeVisible();
  const [proseBounds, recordsBounds] = await Promise.all([
    proposalProse.boundingBox(),
    proposalRecorded.boundingBox(),
  ]);
  expect(proseBounds.y + proseBounds.height).toBeLessThanOrEqual(
    recordsBounds.y + 1,
  );
  await expect(
    proposalRecorded.getByRole("region", {
      name: "Proposal validation",
      exact: true,
    }),
  ).toHaveCount(0);
  const proposalToolDetails = proposalRecorded.getByRole("region", {
    name: "Proposal tool",
    exact: true,
  });
  await expect(proposalToolDetails).toBeVisible();
  await expect(proposalToolDetails).toContainText(
    "Proposal tool metadata remains with the proposal stage.",
  );
  await expect(proposalToolDetails).toContainText(
    "/choice_info/multi_agent/proposal_tool",
  );
  await expect(proposalPhase).not.toContainText("unclassified_note");
  const proposalDetails = proposalRecorded.getByRole("region", {
    name: "Recorded proposal",
    exact: true,
  });
  await expect(proposalDetails).not.toContainText("· recorded");
  await expect(proposalDetails).toContainText(
    "Captured proposal tail: control the observed bleeding before reassessment.",
  );
  await expect(proposalDetails).toContainText(
    "/choice_info/multi_agent/proposal",
  );
  await expectStaticEvidence(proposalPhase);
  await expectNoLineNumbers(proposalDetails);
  const reviewToggle = summary(workflow, "Review");
  await reviewToggle.focus();
  await reviewToggle.press("Enter");
  const reviewPhase = summary(workflow, "Review").locator("..");
  await expect(detailValue(reviewPhase, "Reviewer recommendation")).toHaveText(
    "Assess Patient B before treatment.",
  );
  await expect(detailValue(reviewPhase, "Computed suggestion")).toHaveText(
    "Switch to assessing Patient B.",
  );
  await expect(detailValue(reviewPhase, "Final action")).toHaveCount(0);
  await expect(detailValue(reviewPhase, "Followed guidance")).toHaveCount(0);
  const settings = reviewPhase.getByRole("region", {
    name: "Review settings",
    exact: true,
  });
  const measurements = reviewPhase.getByRole("region", {
    name: "Review measurements",
    exact: true,
  });
  await expect(settings).toBeVisible();
  await expect(detailValue(settings, "Review contract")).toHaveText("advice");
  await expect(detailValue(settings, "Influence")).toHaveText("0.5");
  await expect(detailValue(settings, "Swap threshold")).toHaveText("0.1");
  await expect(settings).not.toContainText("Strength");
  await expect(measurements).toBeVisible();
  await expect(detailValue(measurements, "Confidence")).toHaveText("0.8");
  await expect(detailValue(measurements, "Estimated improvement")).toHaveText(
    "0.4",
  );
  await expect(detailValue(measurements, "Strength")).toHaveText("0.16");
  await expect(measurements).not.toContainText("Influence");
  await expect(
    reviewPhase.getByText(adviceReason, { exact: true }),
  ).toBeVisible();
  const reviewRecorded = reviewPhase.getByRole("region", {
    name: "Review records",
    exact: true,
  });
  const assessmentValidation = reviewRecorded.getByRole("region", {
    name: "Reviewer assessment validation",
    exact: true,
  });
  await expect(assessmentValidation).toContainText(
    "Reviewer assessment validation remains with review.",
  );
  await expect(assessmentValidation).toContainText(
    "/choice_info/multi_agent/review_assessment_validation",
  );
  const weighting = reviewRecorded.getByRole("region", {
    name: "Review weighting",
    exact: true,
  });
  await expect(weighting).toContainText("/choice_info/multi_agent/weighting");
  await expect(weighting).toContainText("followed_guidance");
  const call = reviewRecorded.getByRole("region", {
    name: "Review proposed treatment",
    exact: true,
  });
  await expect(
    call.getByText("System: Review the proposed action", {
      exact: false,
    }),
  ).toBeVisible();
  await expectStaticEvidence(reviewPhase);
  await summary(workflow, "Final selection").click();
  const finalPhase = summary(workflow, "Final selection").locator("..");
  await expect(detailValue(finalPhase, "Final action")).toHaveText(
    "Apply a tourniquet to Patient A.",
  );
  await expect(detailValue(finalPhase, "Followed guidance")).toHaveText(
    /no|false/i,
  );
  await expect(detailValue(finalPhase, "Reviewer recommendation")).toHaveCount(
    0,
  );
  await expect(
    finalPhase.getByText(finalResponse, { exact: true }),
  ).toBeVisible();
  const finalRecorded = finalPhase.getByRole("region", {
    name: "Final selection records",
    exact: true,
  });
  const disposition = finalRecorded.getByRole("region", {
    name: "Final disposition",
    exact: true,
  });
  await expect(disposition).toContainText("kept_proposal");
  await expect(disposition).toContainText(
    "/choice_info/multi_agent/disposition",
  );
  await expectStaticEvidence(finalPhase);
  const supporting = decision.getByRole("region", {
    name: "Supporting evidence",
    exact: true,
  });
  const choiceInfo = summary(supporting, "Choice info");
  const choiceInfoDetails = choiceInfo.locator("..");
  const displayedCode = choiceInfoDetails.locator("pre code");
  await expect(displayedCode).toHaveCount(0);
  await choiceInfo.click();
  await expect(displayedCode).toBeVisible();
  const displayed = (await displayedCode.textContent()).trim();
  const expected = JSON.parse(
    population.traceDetails["tr-example-history-2-1"].spans[0].attributes.find(
      (attribute) => attribute.key === "mlflow.spanOutputs",
    ).value.string_value,
  ).source.choice_info;
  expect(
    JSON.parse(displayed.startsWith("{") ? displayed : `{${displayed}}`),
  ).toEqual(expected);
  await expect(choiceInfoDetails).not.toContainText("recorded_debug_note");
  await expectStaticEvidence(choiceInfoDetails);
};

test("reviewer advice and computed suggestion remain distinct from the executed action", async ({
  page,
  population,
}) => {
  const session = population.decisionSessions[1];
  await page.goto(sessionRoute([session]));
  await closeAssistant(page);
  const decision = await expectCompactDecision(page, page, session, [
    "Proposal",
    "Review",
    "Final selection",
  ]);
  await navigation(page, session)
    .getByRole("button", { name: "Decision 1", exact: true })
    .click();
  await expect(navigation(page, session).getByRole("button")).toHaveCount(3);
  await expectReviewerDistinction(decision, population);
  await expectSelection(page, page, session, 1);
  await expectNoLineNumbers(page);
});

test("graph tools expose paired arguments and results without inventing missing context", async ({
  page,
  population,
}) => {
  const session = population.decisionSessions[2];
  await page.goto(sessionRoute([session]));
  await closeAssistant(page);
  const decision = await expectCompactDecision(page, page, session, [
    "Assessment",
    "Treatment",
  ]);
  const assessment = summary(decision, "Assessment");
  await assessment.click();
  const assessmentGroup = assessment.locator("..");
  await expect(
    assessmentGroup
      .getByRole("region", { name: "Classify observed urgency", exact: true })
      .filter({
        hasNot: page.getByRole("region", { name: "Outputs", exact: true }),
      })
      .getByText("Immediate treatment is indicated by the observed bleeding.", {
        exact: true,
      }),
  ).toBeVisible();
  const assessmentRecorded = assessmentGroup.getByRole("region", {
    name: "Assessment records",
    exact: true,
  });
  const toolDetails = assessmentRecorded.getByRole("region", {
    name: "Inspect patient",
    exact: true,
  });
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
  await expect(
    assessmentRecorded
      .getByRole("region", { name: "Classify observed urgency", exact: true })
      .getByText("The request for this model call was not captured.", {
        exact: true,
      }),
  ).toBeVisible();
  await expectStaticEvidence(assessmentGroup);
  const supporting = decision.getByRole("region", {
    name: "Supporting evidence",
    exact: true,
  });
  const captureGaps = summary(supporting, "Capture gaps");
  await expect(captureGaps).toContainText(
    "The model request before urgency classification was not captured.",
  );
  await captureGaps.click();
  await expect(
    supporting
      .getByRole("region", { name: "Capture gaps", exact: true })
      .getByText(
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
  const decision = await expectCompactDecision(page, page, session, []);
  const process = decision.getByRole("region", {
    name: "Decision process",
    exact: true,
  });
  await expect(
    process.getByRole("heading", { name: "Activity log", exact: true }),
  ).toHaveCount(0);
  await expect(
    process.getByText("Stage attribution unavailable", { exact: true }),
  ).toHaveCount(0);
  await expect(
    process.getByText("Workflow stages were not recorded.", { exact: true }),
  ).toHaveCount(0);
  const activity = process.getByRole("list", {
    name: "Recorded activity",
    exact: true,
  });
  await expect(activity).toHaveJSProperty("tagName", "OL");
  const entries = activity.getByRole("listitem");
  await expect(entries).toHaveCount(4);
  const response = entries.nth(1).getByRole("region", {
    name: "Agent response",
    exact: true,
  });
  await expect(response.getByText("Agent", { exact: true })).toBeVisible();
  await expect(
    response.getByText(
      "Patient A has severe bleeding; inspect the recorded treatment options.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(response.locator("pre")).toHaveCount(0);
  const inspect = summary(entries.nth(2), "inspect_patient");
  const treatments = summary(entries.nth(3), "list_treatments");
  await expect(inspect).toContainText("patient_id=patient-a");
  await expect(treatments).toContainText("patient_id=patient-b");
  await expect(
    inspect.getByText("Review response", { exact: true }),
  ).toHaveCount(0);
  await expect(
    treatments.getByText("Review response", { exact: true }),
  ).toBeVisible();
  await expect(
    activity.getByRole("region", { name: "Log excerpt", exact: true }),
  ).toHaveCount(0);
  await inspect.click();
  const inspectDetails = inspect.locator("..");
  const capturedToolCall = inspectDetails.getByRole("region", {
    name: "Log excerpt",
    exact: true,
  });
  await expect(capturedToolCall.locator("pre code")).toHaveText(
    '[bold]*AGENT TOOL CALL*[/bold]: inspect_patient({"patient_id":"patient-a"})',
  );
  await expect(inspectDetails).toContainText(
    "/choice_info/unassociated_log_excerpts/3",
  );
  await expectStaticEvidence(inspectDetails);
  await expect(
    treatments
      .locator("..")
      .getByRole("region", { name: "Log excerpt", exact: true }),
  ).toHaveCount(0);
  const thinking = summary(entries.nth(0), "Thinking");
  const thinkingBody = thinking.locator("..").locator(":scope > :not(summary)");
  await expect(thinkingBody).not.toBeVisible();
  await thinking.click();
  await expect(thinkingBody).toContainText(
    "Check the observed bleeding against the available treatment options.",
  );
  await expectStaticEvidence(thinking.locator(".."));

  const attributionGap =
    "Some log sections have no verified component owner and are shown without call attribution.";
  const supporting = decision.getByRole("region", {
    name: "Supporting evidence",
    exact: true,
  });
  const captureGaps = summary(supporting, "Capture gaps");
  await expect(captureGaps).toContainText(
    "No intermediate ADM activity can be associated with the final decision record.",
  );
  await captureGaps.click();
  const gapDetails = supporting.getByRole("region", {
    name: "Capture gaps",
    exact: true,
  });
  await expect(
    gapDetails.getByText(
      "No intermediate ADM activity can be associated with the final decision record.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(
    gapDetails.getByText(
      "The unassociated log excerpt cannot establish which model call produced this action.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(gapDetails).not.toContainText(attributionGap);
  const sourceLogs = summary(supporting, "Source logs");
  await expect(sourceLogs).toContainText(/\b1 excerpt\b/);
  await expect(sourceLogs).toContainText("System prompt");
  await sourceLogs.click();
  const logSections = sourceLogs.locator("..");
  await expect(summary(decision, "Recorded decision")).toHaveCount(0);
  await expect(summary(decision, "Other recorded log sections")).toHaveCount(0);
  const recordedLogs = logSections.getByRole("region", {
    name: "Log excerpts",
    exact: true,
  });
  await expect(recordedLogs.locator("pre code")).toHaveText(
    "[bold]*AGENT SYSTEM PROMPT*[/bold]:\nUse the recorded observations to select an available treatment.",
  );
  await expect(recordedLogs).not.toContainText("inspect_patient");
  await expect(recordedLogs).not.toContainText("list_treatments");
  await expectStaticEvidence(logSections);
  await expect(summary(decision, "Legacy decision record")).toHaveCount(0);
  await expect(
    navigation(page, session).getByRole("button", { name: /^Open .* phase$/ }),
  ).toHaveCount(0);
  await expect(
    decision.getByText("Input context was not captured.", { exact: true }),
  ).toHaveCount(0);
  const output = decision.getByRole("region", { name: "Output", exact: true });
  await output.getByRole("button", { name: "See more", exact: true }).click();
  await expect(output).toContainText("Treat Patient A for severe bleeding.");

  await navigation(page, session)
    .getByRole("button", { name: "Decision 2", exact: true })
    .click();
  const secondDecision = card(page, 2);
  const secondSupporting = secondDecision.getByRole("region", {
    name: "Supporting evidence",
    exact: true,
  });
  await expect(summary(secondSupporting, "Source logs")).toHaveCount(0);
  const secondActivity = secondDecision.getByRole("list", {
    name: "Recorded activity",
    exact: true,
  });
  await expect(secondActivity.getByRole("listitem")).toHaveCount(4);
  await expect(
    secondDecision.getByText("Stage attribution unavailable", { exact: true }),
  ).toHaveCount(0);
  await summary(secondSupporting, "Capture gaps").click();
  await expect(
    secondSupporting.getByRole("region", { name: "Capture gaps", exact: true }),
  ).not.toContainText(attributionGap);
  await expectSelection(page, page, session, 2);

  await navigation(page, session)
    .getByRole("button", { name: "Decision 3", exact: true })
    .click();
  const thirdDecision = card(page, 3);
  const thirdSupporting = thirdDecision.getByRole("region", {
    name: "Supporting evidence",
    exact: true,
  });
  await expect(summary(thirdSupporting, "Source logs")).toHaveCount(0);
  await expect(
    thirdDecision.getByRole("list", {
      name: "Recorded activity",
      exact: true,
    }),
  ).toHaveCount(0);
  await summary(thirdSupporting, "Capture gaps").click();
  await expect(
    thirdSupporting
      .getByRole("region", { name: "Capture gaps", exact: true })
      .getByText(attributionGap, { exact: true }),
  ).toBeVisible();
  await expectSelection(page, page, session, 3);
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
    await expectCompactDecision(
      page,
      panes[index],
      sessions[index],
      index === 0
        ? ["State tracking", "Predict urgency"]
        : ["Proposal", "Review", "Final selection"],
    );
    await navigation(page, sessions[index])
      .getByRole("button", { name: "Decision 2", exact: true })
      .click();
  }
  await openPipelineStage(card(panes[0]));
  await expectReviewerDistinction(card(panes[1]), population);
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
