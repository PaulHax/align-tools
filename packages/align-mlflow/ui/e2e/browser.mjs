import { test as base, expect } from "@playwright/test";
import {
  createPopulation,
  createDecisionPopulation,
  installApiFixtures,
} from "./fixtures.mjs";

export const test = base.extend({
  runCount: [8, { option: true }],
  duplicateScores: [false, { option: true }],
  decisionHistory: [false, { option: true }],
  population: async (
    { page, runCount, duplicateScores, decisionHistory },
    use,
  ) => {
    const population = decisionHistory
      ? createDecisionPopulation(runCount)
      : createPopulation(runCount, { duplicateScores });
    const unexpected = await installApiFixtures(page, population);
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.addLocatorHandler(
      page.getByRole("button", { name: "Got it", exact: true }),
      (button) => button.click(),
    );
    await use(population);
    expect(unexpected).toEqual([]);
    expect(errors).toEqual([]);
  },
});

export const openResults = async (page, population) => {
  await page.goto("/#/experiments/1/open-world-results");
  await expect(
    page.getByLabel(
      `Episodes: ${population.traces.filter((trace) => trace.tags.step === "1").length}`,
      { exact: true },
    ),
  ).toBeVisible();
};

export const sectionButton = (page, name) =>
  page.getByRole("button", { name, exact: true });
export { expect };
