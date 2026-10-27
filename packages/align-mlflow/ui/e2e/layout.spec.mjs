import { test, expect, openResults, sectionButton } from "./browser.mjs";

const closeAssistant = async (page) => {
  const close = page
    .locator('[data-assistant-ui="true"]')
    .getByRole("button", { name: "Close", exact: true });
  await close.click();
  await expect(close).not.toBeVisible();
};

const expectSessions = async (page, traces) => {
  const sessions = page.getByRole("region", { name: "Sessions", exact: true });
  await expect(
    sessions.getByRole("region", { name: "Traces", exact: true }),
  ).toHaveAttribute("aria-busy", "false");
  const rows = sessions.getByRole("checkbox", { name: /^Select session / });
  await expect(rows).toHaveCount(traces.length);
  const ids = await rows.evaluateAll((elements) =>
    elements
      .map((element) =>
        element.getAttribute("aria-label").slice("Select session ".length),
      )
      .sort(),
  );
  expect(ids).toEqual(
    traces.map((trace) => trace.trace_metadata["mlflow.trace.session"]).sort(),
  );
  return sessions;
};

test("keeps run identity lines stacked within the compact plot row", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  for (const configuration of [
    population.configurations[0],
    population.configurations[2],
  ]) {
    const identity = page.getByRole("button", {
      name: `${configuration.adm} / ${configuration.model} · ${configuration.name}`,
      exact: true,
    });
    const suffix =
      configuration.alignmentMode === "unaligned" ? " · Unaligned" : "";
    const lines = [
      identity.locator("strong"),
      identity.getByText(configuration.model, { exact: true }),
      identity.getByText(
        configuration.name.slice(configuration.adm.length + 1) + suffix,
        { exact: true },
      ),
    ];
    await expect(identity).toBeVisible();
    const buttonBounds = await identity.boundingBox();
    const contentWidth = await identity.evaluate((element) => {
      const style = getComputedStyle(element);
      return (
        element.clientWidth -
        Number.parseFloat(style.paddingLeft) -
        Number.parseFloat(style.paddingRight)
      );
    });
    expect(contentWidth).toBeGreaterThanOrEqual(170);
    const bounds = await Promise.all(lines.map((line) => line.boundingBox()));
    for (const line of bounds) {
      expect(line.width).toBeGreaterThanOrEqual(contentWidth - 1);
      expect(line.x).toBeGreaterThanOrEqual(buttonBounds.x);
      expect(line.x + line.width).toBeLessThanOrEqual(
        buttonBounds.x + buttonBounds.width + 1,
      );
      expect(line.y).toBeGreaterThanOrEqual(buttonBounds.y - 1);
      expect(line.y + line.height).toBeLessThanOrEqual(
        buttonBounds.y + buttonBounds.height + 1,
      );
    }
    expect(bounds[1].y).toBeGreaterThanOrEqual(
      bounds[0].y + bounds[0].height - 1,
    );
    expect(bounds[2].y).toBeGreaterThanOrEqual(
      bounds[1].y + bounds[1].height - 1,
    );
    expect(buttonBounds.height).toBeLessThanOrEqual(50);
    if (configuration.alignmentMode === "unaligned") {
      const adm = await lines[0].evaluate((element) => ({
        width: element.clientWidth,
        textWidth: element.scrollWidth,
      }));
      expect(adm.textWidth).toBeLessThanOrEqual(adm.width);
    }
  }
});

test("keeps matrix run and case labels inside their columns", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  const table = page.getByRole("table", {
    name: "Open World results matrix",
    exact: true,
  });
  for (const header of (
    await table.getByRole("row").first().getByRole("columnheader").all()
  ).slice(1)) {
    const inspect = header.getByRole("button", {
      name: /^Inspect configuration /,
    });
    const details = header.getByRole("button", { name: /^Details for run / });
    const [cellBounds, inspectBounds, detailBounds] = await Promise.all([
      header.boundingBox(),
      inspect.boundingBox(),
      details.boundingBox(),
    ]);
    expect(inspectBounds.x).toBeGreaterThanOrEqual(cellBounds.x);
    expect(inspectBounds.x + inspectBounds.width).toBeLessThanOrEqual(
      detailBounds.x + 1,
    );
    expect(detailBounds.x + detailBounds.width).toBeLessThanOrEqual(
      cellBounds.x + cellBounds.width + 1,
    );
    const fragments = await inspect
      .locator("span, strong")
      .evaluateAll((elements) =>
        elements.map((element) => element.getBoundingClientRect().right),
      );
    for (const right of fragments)
      expect(right).toBeLessThanOrEqual(
        inspectBounds.x + inspectBounds.width + 1,
      );
  }
  for (const header of await table.getByRole("rowheader").all()) {
    const bounds = await header.boundingBox();
    const labelBounds = await header.getByRole("button").boundingBox();
    expect(labelBounds.x + labelBounds.width).toBeLessThanOrEqual(
      bounds.x + bounds.width + 1,
    );
  }
});

test("uses the existing evaluation runs table and removes duplicate run cards", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  await expect(
    page.getByRole("region", { name: "Run summaries", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText("Performance by run", { exact: true }),
  ).toHaveCount(0);
  const runs = page.getByRole("link", {
    name: "Runs: 8 (evaluation runs)",
    exact: true,
  });
  await expect(runs).toHaveAttribute(
    "href",
    /\/experiments\/1\/evaluation-runs\?workflowType=genai/,
  );
  await runs.click();
  await expect(page).toHaveURL(
    /#\/experiments\/1\/evaluation-runs\?workflowType=genai/,
  );
  await expect(
    page.getByRole("columnheader").filter({ hasText: "ta3_alignment_mean" }),
  ).toBeVisible();
  const runRow = page
    .getByRole("row")
    .filter({ hasText: population.configurations[0].name });
  await expect(runRow).toBeVisible();
  await expect(runRow).toContainText("0.345");
});

test("collapses native sections and restores sessions when inspecting a matrix score", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  for (const name of ["Episode scores", "Scenario/target matrix"]) {
    const button = sectionButton(page, name);
    await expect(button).toHaveAttribute("aria-expanded", "true");
    await button.click();
    await expect(button).toHaveAttribute("aria-expanded", "false");
    await button.click();
    await expect(button).toHaveAttribute("aria-expanded", "true");
  }
  await expect(sectionButton(page, "Sessions (episodes)")).toHaveAttribute(
    "aria-expanded",
    "false",
  );
  const matrix = page.getByRole("table", {
    name: "Open World results matrix",
    exact: true,
  });
  const pair = population.pairs[0];
  const row = matrix.getByRole("row").filter({
    has: page.getByRole("button", {
      name: `Inspect scenario ${pair.scenario} / evaluated target ${pair.target}`,
      exact: true,
    }),
  });
  await row.getByRole("rowheader").getByRole("button").click();
  await expect(sectionButton(page, "Sessions (episodes)")).toHaveAttribute(
    "aria-expanded",
    "true",
  );
  await expectSessions(
    page,
    population.traces.filter((trace) => trace.tags.episode === "1"),
  );
  await sectionButton(page, "Sessions (episodes)").click();
  await row.getByRole("cell").first().getByRole("button").click();
  await expectSessions(page, [population.traces[0]]);
});

test("opens a new session scope on the same mounted page and preserves manual collapse", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  const scope = (configuration) =>
    JSON.stringify([
      {
        field: "metadata",
        key: "mlflow.sourceRun",
        operator: "=",
        value: configuration.id,
      },
    ]);
  const scopedUrl = (configuration) =>
    "/#/experiments/1/open-world-results?resultsSessionFilters=" +
    encodeURIComponent(scope(configuration));
  await page.goto(scopedUrl(population.configurations[0]));
  await expectSessions(page, population.traces.slice(0, 24));
  await sectionButton(page, "Sessions (episodes)").click();
  await page.goto(
    scopedUrl(population.configurations[0]) + "&startTimeLabel=ALL",
  );
  await expect(sectionButton(page, "Sessions (episodes)")).toHaveAttribute(
    "aria-expanded",
    "false",
  );
  await page.goto(scopedUrl(population.configurations[1]));
  await expectSessions(page, population.traces.slice(24, 48));
});

test("keeps matrix rows compact and exposes full run metadata on demand", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  const region = page.getByRole("region", {
    name: "Scenario and target score matrix",
    exact: true,
  });
  const table = region.getByRole("table", {
    name: "Open World results matrix",
    exact: true,
  });
  await expect(table.getByRole("row")).toHaveCount(25);
  const dimensions = await region.evaluate((element) => ({
    height: element.getBoundingClientRect().height,
    scrollHeight: element.scrollHeight,
    clientHeight: element.clientHeight,
  }));
  expect(dimensions.height).toBeLessThanOrEqual(451);
  expect(dimensions.scrollHeight).toBeGreaterThan(dimensions.clientHeight);
  const rowHeights = await table
    .getByRole("row")
    .evaluateAll((rows) =>
      rows.slice(1).map((row) => row.getBoundingClientRect().height),
    );
  expect(Math.max(...rowHeights)).toBeLessThanOrEqual(40);
  await closeAssistant(page);
  await page
    .getByRole("region", { name: "Dashboard scope", exact: true })
    .scrollIntoViewIfNeeded();
  await page.screenshot({
    path: test.info().outputPath("eight-runs-viewport.png"),
  });
  await page.screenshot({
    path: test.info().outputPath("eight-runs-full-page.png"),
    fullPage: true,
  });
  await region.scrollIntoViewIfNeeded();
  await page.screenshot({
    path: test.info().outputPath("eight-runs-matrix.png"),
  });
  const configuration = population.configurations[0];
  await page
    .getByRole("button", {
      name: `Details for run 1: ${configuration.name}`,
      exact: true,
    })
    .click();
  const details = page
    .getByRole("heading", { name: "Run 1 details", exact: true })
    .locator("xpath=..");
  for (const value of [
    configuration.name,
    configuration.adm,
    configuration.model,
    configuration.id,
    configuration.driver,
    configuration.sourceVersion,
    configuration.scoringVersion,
  ]) {
    await expect(details.getByText(value, { exact: true })).toBeVisible();
  }
  await expect(
    details.getByRole("link", { name: "Open run", exact: true }),
  ).toHaveAttribute("href", new RegExp("/runs/" + configuration.id));
});

test("bounds the hydrated native sessions table and retains its filters", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  await sectionButton(page, "Sessions (episodes)").click();
  const sessions = await expectSessions(page, population.traces);
  const bounds = await sessions.boundingBox();
  expect(bounds.height).toBeLessThanOrEqual(641);
  const scrollPanes = await sessions.evaluate((element) =>
    [element, ...element.querySelectorAll("*")]
      .filter(
        (candidate) =>
          /auto|scroll/.test(getComputedStyle(candidate).overflowY) &&
          candidate.scrollHeight > candidate.clientHeight,
      )
      .map((candidate) => candidate.getBoundingClientRect().height),
  );
  expect(scrollPanes.length).toBeGreaterThan(0);
  expect(Math.max(...scrollPanes)).toBeLessThanOrEqual(641);
  await sessions.getByRole("button", { name: "Filters", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Add filter", exact: true }),
  ).toBeVisible();
});

test("deduplicates matching targets while retaining mismatched and unaligned evidence", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  const chart = page.getByRole("region", {
    name: "Episode score distributions",
    exact: true,
  });
  const configuration = population.configurations[1];
  const episodePoint = (run, caseNumber) => {
    const trace = population.traces.find(
      (trace) =>
        trace.trace_metadata["mlflow.sourceRun"] === run.id &&
        trace.tags.episode === String(caseNumber),
    );
    const value = Number(trace.assessments[0].feedback.value.toPrecision(4));
    return chart
      .getByRole("group", {
        name: `Episode scores for ${run.adm} / ${run.model} · ${run.name}`,
        exact: true,
      })
      .getByRole("button", {
        name: new RegExp(
          "^Episode score " + String(value).replace(".", "\\.") + ";",
        ),
      });
  };
  await episodePoint(configuration, 1).hover();
  let tooltip = page.getByRole("tooltip");
  await expect(tooltip).toBeVisible();
  await expect(
    tooltip.getByText(population.pairs[0].target, { exact: true }),
  ).toHaveCount(1);
  await episodePoint(configuration, 2).hover();
  tooltip = page.getByRole("tooltip");
  await expect(tooltip).toBeVisible();
  await expect(
    tooltip.getByText(population.pairs[1].target, { exact: true }),
  ).toBeVisible();
  await expect(
    tooltip.getByText(population.pairs[5].target, { exact: true }),
  ).toBeVisible();
  const baseline = population.configurations[0];
  await episodePoint(baseline, 1).hover();
  await expect(
    page
      .getByRole("tooltip")
      .getByText("Unaligned (no requested target)", { exact: true }),
  ).toBeVisible();
});

test.describe("large comparison", () => {
  test.use({ runCount: 100 });

  test("contains both axes of the matrix and compact plots without widening the document", async ({
    page,
    population,
  }) => {
    await openResults(page, population);
    await closeAssistant(page);
    await expect(sectionButton(page, "Sessions (episodes)")).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    const matrix = page.getByRole("region", {
      name: "Scenario and target score matrix",
      exact: true,
    });
    const table = matrix.getByRole("table", {
      name: "Open World results matrix",
      exact: true,
    });
    await expect(table.getByRole("row")).toHaveCount(25);
    await expect(
      table.getByRole("row").first().getByRole("columnheader"),
    ).toHaveCount(101);
    const bounds = await matrix.evaluate((element) => ({
      height: element.getBoundingClientRect().height,
      width: element.clientWidth,
      scrollWidth: element.scrollWidth,
      scrollHeight: element.scrollHeight,
      clientHeight: element.clientHeight,
    }));
    expect(bounds.height).toBeLessThanOrEqual(451);
    expect(bounds.scrollWidth).toBeGreaterThan(bounds.width);
    expect(bounds.scrollHeight).toBeGreaterThan(bounds.clientHeight);
    const documentBounds = await page.evaluate(() => ({
      width: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
      height: window.innerHeight,
      scrollHeight: document.documentElement.scrollHeight,
    }));
    expect(documentBounds.scrollWidth).toBeLessThanOrEqual(
      documentBounds.width + 1,
    );
    expect(documentBounds.scrollHeight).toBeLessThanOrEqual(
      documentBounds.height * 2,
    );
    const plots = page.getByRole("region", {
      name: "Episode score plots",
      exact: true,
    });
    await expect(
      plots.getByRole("button", { name: /^Episode score / }),
    ).toHaveCount(2400);
    const plotBounds = await plots.evaluate((element) => ({
      height: element.getBoundingClientRect().height,
      scrollHeight: element.scrollHeight,
      clientHeight: element.clientHeight,
    }));
    expect(plotBounds.height).toBeLessThanOrEqual(641);
    expect(plotBounds.scrollHeight).toBeGreaterThan(plotBounds.clientHeight);
    await page.screenshot({
      path: test.info().outputPath("hundred-runs-viewport.png"),
    });
    await page.screenshot({
      path: test.info().outputPath("hundred-runs-full-page.png"),
      fullPage: true,
    });
    await matrix.scrollIntoViewIfNeeded();
    await page.screenshot({
      path: test.info().outputPath("hundred-runs-matrix.png"),
    });
  });
});
