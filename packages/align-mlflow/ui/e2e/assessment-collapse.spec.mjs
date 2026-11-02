import { test, expect } from "./browser.mjs";

test.use({
  decisionHistory: true,
  viewport: { width: 960, height: 1000 },
});

test("session assessments collapse without losing the selected decision or a notes draft", async ({
  page,
  population,
}) => {
  const session = population.decisionSessions[1];
  await page.goto(
    "/#/experiments/1/chat-sessions/" + encodeURIComponent(session),
  );
  const closeAssistant = page
    .locator('[data-assistant-ui="true"]')
    .getByRole("button", { name: "Close", exact: true });
  await expect(closeAssistant).toBeVisible();
  await closeAssistant.click();
  await expect(closeAssistant).not.toBeVisible();
  const navigation = page.getByRole("navigation", {
    name: `${session} decisions`,
    exact: true,
  });
  await expect(navigation).toBeVisible();
  await expect(navigation.getByRole("button")).toHaveText([
    "Decision 1",
    "Decision 2",
    "Decision 3",
  ]);
  const selectedNavigation = navigation.getByRole("button", {
    name: "Decision 2",
    exact: true,
  });
  await selectedNavigation.click();
  const selectedDecision = page.getByRole("group", {
    name: "Decision 2",
    exact: true,
  });
  const selectedTitle = selectedDecision.getByRole("button", {
    name: "Decision 2",
    exact: true,
  });
  await expect(selectedNavigation).toHaveAttribute("aria-pressed", "true");
  await expect(selectedTitle).toHaveAttribute("aria-pressed", "true");

  const assessments = page.getByTestId("assessments-pane");
  await expect(assessments).toBeVisible();
  await expect(
    assessments.getByText("ta3_session_alignment_score", { exact: true }),
  ).toBeVisible();
  await expect(assessments.getByText("0.03", { exact: true })).toBeVisible();
  const notes = assessments.getByRole("textbox");
  const draft = "Review the proposal before comparing the next decision.";
  await notes.fill(draft);
  await expect(
    assessments.getByRole("button", { name: "Save", exact: true }),
  ).toBeEnabled();
  const expandedWidth = (await selectedDecision.boundingBox()).width;

  const close = page.getByRole("button", {
    name: "Collapse session assessments",
    exact: true,
  });
  const togglePosition = await close.boundingBox();
  await close.focus();
  await page.keyboard.press("Enter");
  await expect(assessments).not.toBeVisible();
  const reopen = page.getByRole("button", {
    name: "Expand session assessments",
    exact: true,
  });
  await expect(reopen).toBeFocused();
  await expect(reopen).toHaveAttribute("aria-expanded", "false");
  const collapsedPosition = await reopen.boundingBox();
  expect(Math.abs(collapsedPosition.x - togglePosition.x)).toBeLessThanOrEqual(
    1,
  );
  expect(Math.abs(collapsedPosition.y - togglePosition.y)).toBeLessThanOrEqual(
    1,
  );
  await expect(
    page.getByRole("button", {
      name: /^(Collapse|Expand) session assessments$/,
    }),
  ).toHaveCount(1);
  await expect(navigation).toBeVisible();
  await expect(selectedNavigation).toHaveAttribute("aria-pressed", "true");
  await expect(selectedTitle).toHaveAttribute("aria-pressed", "true");
  expect(
    (await selectedDecision.boundingBox()).width - expandedWidth,
  ).toBeGreaterThanOrEqual(250);

  await reopen.click();
  await expect(assessments).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: "Collapse session assessments",
      exact: true,
    }),
  ).toHaveAttribute("aria-expanded", "true");
  await expect(assessments.getByText("0.03", { exact: true })).toBeVisible();
  await expect(notes).toHaveValue(draft);
  await expect(
    assessments.getByRole("button", { name: "Save", exact: true }),
  ).toBeEnabled();
  await expect(selectedNavigation).toHaveAttribute("aria-pressed", "true");
  await expect(selectedTitle).toHaveAttribute("aria-pressed", "true");
});
