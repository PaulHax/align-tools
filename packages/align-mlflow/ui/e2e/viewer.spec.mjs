import { test, expect } from "./browser.mjs";

test.use({ decisionHistory: true });

const names = ["Decision 1", "Decision 2", "Decision 3"];
const sessionRoute = (sessions) => {
  const query = new URLSearchParams();
  sessions
    .slice(1)
    .forEach((session) => query.append("compareSessionId", session));
  return (
    "/#/experiments/1/chat-sessions/" +
    encodeURIComponent(sessions[0]) +
    (query.size ? "?" + query : "")
  );
};
const sidebar = (page, session) =>
  page.getByRole("navigation", { name: `${session} decisions`, exact: true });
const card = (scope, number) =>
  scope.getByRole("group", { name: `Decision ${number}`, exact: true });
const captureViewer = async (page, scope, selected, filename) => {
  const close = page
    .locator('[data-assistant-ui="true"]')
    .getByRole("button", { name: "Close", exact: true });
  await close.click();
  await expect(close).not.toBeVisible();
  await card(scope, selected)
    .getByRole("button", { name: `Decision ${selected}`, exact: true })
    .scrollIntoViewIfNeeded();
  await page.mouse.move(0, 0);
  await page.screenshot({ path: test.info().outputPath(filename) });
};
const expectOrder = async (scope, navigation) => {
  await expect(navigation.getByRole("button")).toHaveText(names);
  await expect(
    scope.getByRole("group", { name: /^Decision [123]$/ }),
  ).toHaveCount(3);
  const cards = await scope
    .getByRole("group", { name: /^Decision [123]$/ })
    .evaluateAll((elements) =>
      elements.map((element) => element.getAttribute("aria-label")),
    );
  expect(cards).toEqual(names);
};
const expectSelection = async (scope, navigation, selected) => {
  for (let number = 1; number <= 3; number++) {
    const value = number === selected ? "true" : "false";
    await expect(
      navigation.getByRole("button", {
        name: `Decision ${number}`,
        exact: true,
      }),
    ).toHaveAttribute("aria-pressed", value);
    await expect(
      card(scope, number).getByRole("button", {
        name: `Decision ${number}`,
        exact: true,
      }),
    ).toHaveAttribute("aria-pressed", value);
  }
};
const selectCardSurface = async (page, scope, navigation, number) => {
  const target = card(scope, number);
  await target.scrollIntoViewIfNeeded();
  const message = target.getByText(new RegExp(`visible-[123]-${number}`));
  await message.evaluate((element) =>
    element.scrollIntoView({
      block: "center",
      inline: "nearest",
      behavior: "instant",
    }),
  );
  await expect(message).toBeInViewport();
  const scrollPosition = () =>
    target.evaluate((element) => {
      for (
        let parent = element.parentElement;
        parent;
        parent = parent.parentElement
      ) {
        if (
          /auto|scroll/.test(getComputedStyle(parent).overflowY) &&
          parent.scrollHeight > parent.clientHeight + 1
        )
          return parent.scrollTop;
      }
      throw new Error("Expected the complete decision history to scroll");
    });
  const before = await scrollPosition();
  expect(before).toBeGreaterThan(0);
  const [bounds, messageBounds] = await Promise.all([
    target.boundingBox(),
    message.boundingBox(),
  ]);
  await page.mouse.click(
    bounds.x + 8,
    messageBounds.y + messageBounds.height / 2,
  );
  await expectSelection(scope, navigation, number);
  await test.info().attach("card-surface-viewport", {
    body: JSON.stringify({
      before,
      after: await scrollPosition(),
      messageBefore: messageBounds,
      messageAfter: await message.boundingBox(),
      history: await target.evaluate((element) => ({
        viewport: element.parentElement.getBoundingClientRect().toJSON(),
        contentHeight: element.parentElement.scrollHeight,
        clientHeight: element.parentElement.clientHeight,
      })),
    }),
    contentType: "application/json",
  });
  await expect(message).toBeInViewport();
  expect(await scrollPosition()).toBe(before);
};

const togglePayloadControls = async (scope, navigation, selected) => {
  const unselected = card(scope, selected === 1 ? 2 : 1);
  await expect(
    unselected.getByText("Show 1 more", { exact: true }),
  ).toHaveCount(2);
  for (const name of ["Inputs", "Outputs"]) {
    const toggle = unselected
      .getByRole("heading", { name, exact: true })
      .locator("..")
      .getByRole("button");
    await toggle.click();
    await expect(
      unselected.getByText("Show 1 more", { exact: true }),
    ).toHaveCount(1);
    await expectSelection(scope, navigation, selected);
    await toggle.click();
    await expect(
      unselected.getByText("Show 1 more", { exact: true }),
    ).toHaveCount(2);
  }
};

test("single session shares decision ordering and selection across cards and sidebar", async ({
  page,
  population,
}) => {
  const [firstSession, secondSession] = population.decisionSessions;
  await page.goto(sessionRoute([firstSession]));
  const navigation = sidebar(page, firstSession);
  await expectOrder(page, navigation);
  await expectSelection(page, navigation, undefined);
  await card(page, 3).hover();
  await expectSelection(page, navigation, undefined);
  await selectCardSurface(page, page, navigation, 3);
  await togglePayloadControls(page, navigation, 3);
  await navigation
    .getByRole("button", { name: "Decision 2", exact: true })
    .click();
  await expectSelection(page, navigation, 2);
  const firstCardButton = card(page, 1).getByRole("button", {
    name: "Decision 1",
    exact: true,
  });
  await firstCardButton.focus();
  await page.keyboard.press("Space");
  await expectSelection(page, navigation, 1);
  await card(page, 2).hover();
  await card(page, 2)
    .getByRole("button", { name: "View full trace", exact: true })
    .click();
  const drawer = page.getByRole("dialog");
  await expect(drawer).toBeVisible();
  await drawer
    .getByRole("button", { name: "Close trace panel", exact: true })
    .click();
  await expect(drawer).not.toBeVisible();
  await expectSelection(page, navigation, 1);
  await page.goto(sessionRoute([secondSession]));
  const nextNavigation = sidebar(page, secondSession);
  await expectOrder(page, nextNavigation);
  await expectSelection(page, nextNavigation, undefined);
  await selectCardSurface(page, page, nextNavigation, 2);
  await captureViewer(page, page, 2, "single-session-selection.png");
});

for (const count of [2, 3]) {
  test(`${count} session panes keep independent chosen decisions and compare their exact trace IDs`, async ({
    page,
    population,
  }) => {
    const sessions = population.decisionSessions.slice(0, count);
    await page.goto(sessionRoute(sessions));
    const panes = sessions.map((session) =>
      page.getByRole("region", { name: session, exact: true }),
    );
    const navigations = sessions.map((session) => sidebar(page, session));
    for (let index = 0; index < count; index++) {
      await expectOrder(panes[index], navigations[index]);
      await expectSelection(panes[index], navigations[index], undefined);
    }
    await expect(
      page.getByText("0 decisions selected", { exact: true }),
    ).toBeVisible();
    const compare = page.getByRole("button", {
      name: "Compare selected decisions",
      exact: true,
    });
    await expect(compare).toBeDisabled();
    await card(panes[0], 2).hover();
    await expect(
      page.getByText("0 decisions selected", { exact: true }),
    ).toBeVisible();
    await selectCardSurface(page, panes[0], navigations[0], 3);
    await navigations[1]
      .getByRole("button", { name: "Decision 1", exact: true })
      .click();
    await expectSelection(panes[1], navigations[1], 1);
    if (count === 3) {
      await card(panes[2], 2)
        .getByRole("button", { name: "Decision 2", exact: true })
        .focus();
      await page.keyboard.press("Enter");
      await expectSelection(panes[2], navigations[2], 2);
    }
    await expect(
      page.getByText(`${count} decisions selected`, { exact: true }),
    ).toBeVisible();
    await expect(compare).toBeEnabled();
    await card(panes[0], 1).hover();
    await expectSelection(panes[0], navigations[0], 3);
    await togglePayloadControls(panes[0], navigations[0], 3);
    await card(panes[0], 1)
      .getByRole("button", { name: "Decision 1", exact: true })
      .focus();
    await page.keyboard.press("Space");
    const selectedSteps = [1, 1, 2];
    for (let index = 0; index < count; index++)
      await expectSelection(
        panes[index],
        navigations[index],
        selectedSteps[index],
      );
    await expect(
      page.getByText(`${count} decisions selected`, { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", {
        name: "Clear selected decision",
        exact: true,
      }),
    ).toHaveCount(0);
    await captureViewer(
      page,
      panes[0],
      selectedSteps[0],
      `comparison-${count}-selected-decisions.png`,
    );
    await compare.click();
    const drawer = page.getByRole("dialog", {
      name: "Comparing selected decisions",
      exact: true,
    });
    await expect(drawer).toBeVisible();
    for (let index = 0; index < count; index++) {
      const selectedId = `tr-example-history-${index + 1}-${selectedSteps[index]}`;
      await expect(
        drawer.getByText(selectedId, { exact: true }).first(),
      ).toBeVisible();
      for (let step = 1; step <= 3; step++) {
        if (step !== selectedSteps[index])
          await expect(
            drawer.getByText(`tr-example-history-${index + 1}-${step}`, {
              exact: true,
            }),
          ).toHaveCount(0);
      }
    }
  });
}
