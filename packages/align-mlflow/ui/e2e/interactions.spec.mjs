import { test, expect, openResults, sectionButton } from "./browser.mjs";

const runPlot = (page, configuration) =>
  page.getByRole("group", {
    name: `Episode scores for ${configuration.adm} / ${configuration.model} · ${configuration.name}`,
    exact: true,
  });

const expectSessions = async (page, ids) => {
  await expect(sectionButton(page, "Sessions (episodes)")).toHaveAttribute(
    "aria-expanded",
    "true",
  );
  const sessions = page.getByRole("region", { name: "Sessions", exact: true });
  await expect(
    sessions.getByRole("region", { name: "Traces", exact: true }),
  ).toHaveAttribute("aria-busy", "false");
  await expect(
    sessions.getByRole("checkbox", { name: /^Select session / }),
  ).toHaveCount(ids.length);
  for (const id of ids)
    await expect(
      sessions.getByRole("checkbox", {
        name: `Select session ${id}`,
        exact: true,
      }),
    ).toBeVisible();
};

test.describe("counted episode markers", () => {
  test.use({ duplicateScores: true });

  for (const [activation, runIndex] of [
    ["click", 1],
    ["Enter", 0],
    ["Space", 1],
  ]) {
    test(`${activation} selects exactly the identical scores and keeps nearby scores separate`, async ({
      page,
      population,
    }) => {
      const configuration = population.configurations[runIndex];
      const traces = population.traces.filter(
        (trace) =>
          trace.trace_metadata["mlflow.sourceRun"] === configuration.id,
      );
      if (runIndex === 0) {
        traces[0].assessments[0].feedback.value = 0.5;
        traces[1].assessments[0].feedback.value = 0.5;
        traces[2].assessments[0].feedback.value = 0.5001;
        population.runs[runIndex].data.metrics.find(
          (metric) => metric.key === "ta3_alignment_mean",
        ).value =
          traces.reduce(
            (sum, trace) => sum + trace.assessments[0].feedback.value,
            0,
          ) / traces.length;
      }
      const expectedIds = traces
        .slice(0, 2)
        .map((trace) => trace.trace_metadata["mlflow.trace.session"])
        .sort();
      await openResults(page, population);
      const plot = runPlot(page, configuration);
      const counted = plot.getByRole("button", {
        name: /^Episode score 0\.5; 2 episodes;/,
      });
      const nearby = plot.getByRole("button", {
        name: /^Episode score 0\.5001;/,
      });
      await expect(counted).toHaveCount(1);
      await expect(counted.locator("text")).toHaveText("2");
      await expect(counted.locator(".episode-marker")).toHaveAttribute(
        "r",
        "10",
      );
      await expect(nearby).toHaveCount(1);
      await expect(nearby.locator(".episode-marker")).toHaveAttribute("r", "6");
      expect(
        await counted.locator(".episode-marker").getAttribute("cy"),
      ).not.toBe(await nearby.locator(".episode-marker").getAttribute("cy"));
      if (activation === "click") await counted.click();
      else {
        await counted.focus();
        await page.keyboard.press(activation);
      }
      await expect(page).toHaveURL((url) => {
        const parameters = new URLSearchParams(url.hash.split("?")[1]);
        return (
          JSON.stringify(
            JSON.parse(parameters.get("resultsEpisodes") || "[]").sort(),
          ) === JSON.stringify(expectedIds)
        );
      });
      await expectSessions(page, expectedIds);
      await nearby.focus();
      await page.keyboard.press("Enter");
      await expectSessions(page, [
        traces[2].trace_metadata["mlflow.trace.session"],
      ]);
    });
  }
});

test("keeps focused plot and matrix targets visible in the main page", async ({
  page,
  population,
}) => {
  await openResults(page, population);
  const plots = page.getByRole("region", {
    name: "Episode score plots",
    exact: true,
  });
  const point = plots
    .getByRole("group", { name: /^Episode scores for / })
    .last()
    .getByRole("button", { name: /^Episode score / })
    .last();
  await point.focus();
  await point.scrollIntoViewIfNeeded();
  await point.hover();
  await expect(point).toBeFocused();
  await expect(point).toBeInViewport();
  expect(await plots.evaluate((element) => element.scrollTop)).toBe(0);
  const matrix = page.getByRole("region", {
    name: "Scenario and target score matrix",
    exact: true,
  });
  const table = matrix.getByRole("table", {
    name: "Open World results matrix",
    exact: true,
  });
  const lastRow = table.getByRole("row").last();
  const score = lastRow.getByRole("cell").last().getByRole("button");
  await score.focus();
  await score.scrollIntoViewIfNeeded();
  await score.hover();
  await expect(score).toBeFocused();
  await expect(score).toBeInViewport();
  const caseHeader = lastRow.getByRole("rowheader");
  await expect(caseHeader).toBeInViewport();
  const caseBounds = await caseHeader.boundingBox();
  const scoreBounds = await score.boundingBox();
  expect(scoreBounds.x).toBeGreaterThanOrEqual(caseBounds.x + caseBounds.width);
  const offsets = await matrix.evaluate((element) => ({
    top: element.scrollTop,
    left: element.scrollLeft,
  }));
  expect(offsets.top).toBe(0);
  expect(offsets.left).toBeGreaterThan(0);
});

test.describe("responsive plots", () => {
  test.use({ duplicateScores: true });

  test("resizes the score axis while preserving marker lanes and row heights", async ({
    page,
    population,
  }) => {
    await openResults(page, population);
    const plots = page.getByRole("region", {
      name: "Episode score plots",
      exact: true,
    });
    const axis = plots.getByRole("img", {
      name: "Alignment score axis",
      exact: true,
    });
    const plot = runPlot(page, population.configurations[0]);
    const marker = plot
      .getByRole("button", { name: /^Episode score 0;/ })
      .locator(".episode-marker");
    const middle = plot
      .getByRole("button", { name: /^Episode score 0\.3;/ })
      .locator(".episode-marker");
    const counted = runPlot(page, population.configurations[1]).getByRole(
      "button",
      { name: /^Episode score 0\.5; 2 episodes;/ },
    );
    const nearby = runPlot(page, population.configurations[1]).getByRole(
      "button",
      { name: /^Episode score 0\.5001;/ },
    );
    const measure = async () => {
      await page.waitForFunction(() => {
        const axis = document.querySelector(
          'svg[aria-label="Alignment score axis"]',
        );
        return (
          axis &&
          axis.viewBox.baseVal.width ===
            Math.round(axis.parentElement.getBoundingClientRect().width)
        );
      });
      const measuredWidth = Math.round(
        (await axis.locator("..").boundingBox()).width,
      );
      await expect(axis).toHaveAttribute("viewBox", `0 0 ${measuredWidth} 36`);
      await expect(plot).toHaveAttribute(
        "viewBox",
        new RegExp(`^0 0 ${measuredWidth} `),
      );
      expect(await axis.getAttribute("preserveAspectRatio")).not.toBe("none");
      expect(await plot.getAttribute("preserveAspectRatio")).not.toBe("none");
      await expect(counted).toHaveCount(1);
      await expect(counted.locator("text")).toHaveText("2");
      await expect(nearby).toHaveCount(1);
      for (const circle of [marker, counted.locator(".episode-marker")]) {
        const bounds = await circle.boundingBox();
        const dimensions = await circle.evaluate((element) => ({
          diameter: element.r.baseVal.value * 2,
          strokeWidth: Number.parseFloat(getComputedStyle(element).strokeWidth),
        }));
        expect(Math.abs(bounds.width - bounds.height)).toBeLessThan(0.1);
        expect(bounds.width).toBeGreaterThanOrEqual(dimensions.diameter);
        expect(bounds.width).toBeLessThanOrEqual(
          dimensions.diameter + dimensions.strokeWidth + 0.1,
        );
      }
      const rows = await plots
        .getByRole("group", { name: /^Episode scores for / })
        .evaluateAll((elements) =>
          elements
            .map((element) => ({
              name: element.getAttribute("aria-label"),
              svgHeight: element.getBoundingClientRect().height,
              rowHeight: element.parentElement.getBoundingClientRect().height,
              viewBoxHeight: element.viewBox.baseVal.height,
              points: [...element.querySelectorAll('g[role="button"]')]
                .filter((point) =>
                  point.getAttribute("aria-label").startsWith("Episode score "),
                )
                .map((point) => ({
                  name: point.getAttribute("aria-label"),
                  cy: point.querySelector(".episode-marker").getAttribute("cy"),
                  count: Number(
                    point
                      .getAttribute("aria-label")
                      .match(/; (\d+) episodes;/)?.[1] || 1,
                  ),
                }))
                .sort((left, right) => left.name.localeCompare(right.name)),
            }))
            .sort((left, right) => left.name.localeCompare(right.name)),
        );
      expect(rows).toHaveLength(8);
      expect(
        rows
          .flatMap((row) => row.points)
          .reduce((sum, point) => sum + point.count, 0),
      ).toBe(192);
      for (const row of rows) expect(row.svgHeight).toBe(row.viewBoxHeight);
      const dimensions = await plot.evaluate((element) => ({
        width: element.getBoundingClientRect().width,
        viewBox: element.viewBox.baseVal.width,
      }));
      expect(
        Math.abs(dimensions.width - dimensions.viewBox),
      ).toBeLessThanOrEqual(1);
      return {
        width: measuredWidth,
        middleX: Number(await middle.getAttribute("cx")),
        rows,
      };
    };
    const states = [];
    for (const width of [1314, 1800, 1000]) {
      await page.setViewportSize({ width, height: 1000 });
      states.push(await measure());
    }
    expect(states[1].width).toBeGreaterThan(states[0].width);
    expect(states[2].width).toBeLessThan(states[1].width);
    expect(states[1].middleX).toBeGreaterThan(states[0].middleX);
    expect(states[2].middleX).toBeLessThan(states[1].middleX);
    for (const state of states.slice(1))
      expect(state.rows).toEqual(states[0].rows);
    await page.setViewportSize({ width: 1314, height: 1000 });
    const close = page
      .locator('[data-assistant-ui="true"]')
      .getByRole("button", { name: "Close", exact: true });
    await close.click();
    await expect(close).not.toBeVisible();
    const withoutPanel = await measure();
    expect(withoutPanel.width).toBeGreaterThan(states[0].width);
    expect(withoutPanel.rows).toEqual(states[0].rows);
    await page.setViewportSize({ width: 1243, height: 1000 });
    const userWidth = await measure();
    expect(userWidth.rows).toEqual(states[0].rows);
    await page
      .getByRole("heading", { name: "Results overview", exact: true })
      .evaluate((element) =>
        element.scrollIntoView({
          block: "start",
          inline: "nearest",
          behavior: "instant",
        }),
      );
    await page.mouse.move(0, 0);
    await page.screenshot({
      path: test.info().outputPath("header-plots-1243.png"),
    });
  });
});
