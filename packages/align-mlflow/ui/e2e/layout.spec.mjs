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

test("shows complete ADM names inside wide matrix columns without ordinal labels", async ({
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
    const details = header.getByRole("button", { name: /^Details for / });
    const [cellBounds, inspectBounds, detailBounds] = await Promise.all([
      header.boundingBox(),
      inspect.boundingBox(),
      details.boundingBox(),
    ]);
    expect(cellBounds.width).toBeGreaterThanOrEqual(239);
    const detailsLabel = await details.getAttribute("aria-label");
    const configuration = population.configurations.find(
      (item) => `Details for ${item.name}` === detailsLabel,
    );
    expect(configuration).toBeTruthy();
    const adm = inspect.getByText(configuration.adm, { exact: true });
    await expect(adm).toBeVisible();
    const textBounds = await adm.evaluate((element) => ({
      textWidth: element.scrollWidth,
      width: element.clientWidth,
    }));
    expect(textBounds.textWidth).toBeLessThanOrEqual(textBounds.width + 1);
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
  await expect(table.getByText(/^Run \d+$/, { exact: true })).toHaveCount(0);
  for (const header of await table.getByRole("rowheader").all()) {
    const bounds = await header.boundingBox();
    const labelBounds = await header.getByRole("button").boundingBox();
    expect(labelBounds.x + labelBounds.width).toBeLessThanOrEqual(
      bounds.x + bounds.width + 1,
    );
  }
});

test("keeps section explanations in independent header help popovers", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  const explanations = [
    [
      "Results overview",
      /Scored pairs use the scenario and evaluated target reported by TA3\./,
    ],
    ["Episode scores", /Dots show recorded episode scores;/],
    [
      "Scenario/target matrix",
      /Select a score, case, or run to inspect its sessions\./,
    ],
    ["Sessions (episodes)", /Charts and matrix selections filter this table\./],
  ];
  for (const [name, content] of explanations) {
    const header = sectionButton(page, name);
    const expanded = name === "Sessions (episodes)" ? "false" : "true";
    const explanation = page.getByText(content);
    await expect(explanation).not.toBeVisible();
    const help = page.getByRole("button", {
      name: `About ${name}`,
      exact: true,
    });
    await help.focus();
    await expect(header).toHaveAttribute("aria-expanded", expanded);
    const [headerBounds, helpBounds] = await Promise.all([
      header.boundingBox(),
      help.boundingBox(),
    ]);
    expect(helpBounds.y).toBeGreaterThanOrEqual(headerBounds.y - 1);
    expect(helpBounds.y + helpBounds.height).toBeLessThanOrEqual(
      headerBounds.y + headerBounds.height + 1,
    );
    await page.keyboard.press("Enter");
    await expect(explanation).toBeVisible();
    await expect(header).toHaveAttribute("aria-expanded", expanded);
    await page.keyboard.press("Escape");
    await expect(explanation).not.toBeVisible();
    await help.click();
    await expect(explanation).toBeVisible();
    await expect(header).toHaveAttribute("aria-expanded", expanded);
    await page.keyboard.press("Escape");
    await expect(explanation).not.toBeVisible();
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
  for (const name of [
    "Results overview",
    "Episode scores",
    "Scenario/target matrix",
  ]) {
    const button = sectionButton(page, name);
    await expect(button).toHaveAttribute("aria-expanded", "true");
    await expect(button.getByText("Collapse", { exact: true })).toBeVisible();
    await button.click();
    await expect(button).toHaveAttribute("aria-expanded", "false");
    await expect(button.getByText("Expand", { exact: true })).toBeVisible();
    await button.click();
    await expect(button).toHaveAttribute("aria-expanded", "true");
  }
  await expect(sectionButton(page, "Sessions (episodes)")).toHaveAttribute(
    "aria-expanded",
    "false",
  );
  await expect(
    sectionButton(page, "Sessions (episodes)").getByText("Expand", {
      exact: true,
    }),
  ).toBeVisible();
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
  await expect(
    page.getByRole("button", { name: "Back to results", exact: true }),
  ).toHaveCount(0);
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

test("expands the complete matrix naturally and exposes full run metadata on demand", async ({
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
  expect(dimensions.height).toBeGreaterThan(700);
  expect(dimensions.scrollHeight).toBeLessThanOrEqual(
    dimensions.clientHeight + 1,
  );
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
  await page
    .getByRole("heading", { name: "Scenario/target matrix", exact: true })
    .evaluate((element) =>
      element.scrollIntoView({
        block: "start",
        inline: "nearest",
        behavior: "instant",
      }),
    );
  await page.screenshot({
    path: test.info().outputPath("eight-runs-matrix.png"),
  });
  const configuration = population.configurations[0];
  await page
    .getByRole("button", {
      name: `Details for ${configuration.name}`,
      exact: true,
    })
    .click();
  const details = page
    .getByRole("heading", { name: `${configuration.adm} details`, exact: true })
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

test("expands the hydrated native sessions table without nested vertical scrolling", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  await sectionButton(page, "Sessions (episodes)").click();
  const sessions = await expectSessions(page, population.traces);
  const bounds = await sessions.boundingBox();
  expect(bounds.height).toBeGreaterThan(1000);
  const verticalScrollers = await sessions.evaluate((element) =>
    [element, ...element.querySelectorAll("*")]
      .filter(
        (candidate) =>
          /auto|scroll/.test(getComputedStyle(candidate).overflowY) &&
          candidate.scrollHeight > candidate.clientHeight + 1,
      )
      .map((candidate) => ({
        tag: candidate.tagName,
        height: candidate.clientHeight,
        contentHeight: candidate.scrollHeight,
      })),
  );
  expect(verticalScrollers).toEqual([]);
  await expect(
    page.getByRole("button", { name: "Back to results", exact: true }),
  ).toHaveCount(0);
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

  test("expands in the main page and collapses large sections with the keyboard", async ({
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
    const matrixBounds = await matrix.evaluate((element) => ({
      height: element.getBoundingClientRect().height,
      width: element.clientWidth,
      scrollWidth: element.scrollWidth,
      scrollHeight: element.scrollHeight,
      clientHeight: element.clientHeight,
    }));
    expect(matrixBounds.height).toBeGreaterThan(700);
    expect(matrixBounds.scrollWidth).toBeGreaterThanOrEqual(24200);
    expect(matrixBounds.scrollWidth).toBeGreaterThan(matrixBounds.width);
    expect(matrixBounds.scrollHeight).toBeLessThanOrEqual(
      matrixBounds.clientHeight + 1,
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
    expect(plotBounds.height).toBeGreaterThan(5000);
    expect(plotBounds.scrollHeight).toBeLessThanOrEqual(
      plotBounds.clientHeight + 1,
    );
    const documentBounds = await page.evaluate(() => ({
      width: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
    }));
    expect(documentBounds.scrollWidth).toBeLessThanOrEqual(
      documentBounds.width + 1,
    );
    const flowHeight = (button) =>
      button.evaluate((element) => {
        for (
          let parent = element.parentElement;
          parent;
          parent = parent.parentElement
        ) {
          if (/auto|scroll/.test(getComputedStyle(parent).overflowY))
            return parent.scrollHeight;
        }
        return document.documentElement.scrollHeight;
      });
    const expandedHeight = await flowHeight(
      sectionButton(page, "Episode scores"),
    );
    expect(expandedHeight).toBeGreaterThan(5000);
    await sectionButton(page, "Results overview").scrollIntoViewIfNeeded();
    await page.screenshot({
      path: test.info().outputPath("hundred-runs-viewport.png"),
    });
    await page.screenshot({
      path: test.info().outputPath("hundred-runs-full-page.png"),
      fullPage: true,
    });
    await page
      .getByRole("heading", { name: "Scenario/target matrix", exact: true })
      .evaluate((element) =>
        element.scrollIntoView({
          block: "start",
          inline: "nearest",
          behavior: "instant",
        }),
      );
    await page.screenshot({
      path: test.info().outputPath("hundred-runs-matrix.png"),
    });
    for (const [name, key] of [
      ["Episode scores", "Space"],
      ["Scenario/target matrix", "Enter"],
      ["Results overview", "Space"],
    ]) {
      const button = sectionButton(page, name);
      await button.focus();
      await expect(button).toBeInViewport();
      await page.keyboard.press(key);
      await expect(button).toHaveAttribute("aria-expanded", "false");
      await expect(button.getByText("Expand", { exact: true })).toBeVisible();
      const regionName =
        name === "Episode scores"
          ? "Episode score plots"
          : name === "Scenario/target matrix"
            ? "Scenario and target score matrix"
            : "Dashboard scope";
      await expect(
        page.getByRole("region", {
          name: regionName,
          exact: true,
          includeHidden: true,
        }),
      ).not.toBeVisible();
    }
    const collapsedHeight = await flowHeight(
      sectionButton(page, "Episode scores"),
    );
    expect(collapsedHeight).toBeLessThan(expandedHeight / 3);
    for (const name of [
      "Results overview",
      "Episode scores",
      "Scenario/target matrix",
      "Sessions (episodes)",
    ]) {
      await expect(sectionButton(page, name)).toBeInViewport();
    }
    await sectionButton(page, "Scenario/target matrix").focus();
    await page.keyboard.press("Enter");
    await expect(sectionButton(page, "Scenario/target matrix")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    await expect(table.getByRole("row")).toHaveCount(25);
  });
});
