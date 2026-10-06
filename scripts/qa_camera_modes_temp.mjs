export default async function run(page, ui) {
    const before = await ui.snapshot();
    if (!before.includes('tab "Live Camera"')) return { error: "Live Camera tab was not found", snapshot: before };
    await page.locator('[role="tab"][data-key="1"]').evaluate((element) => element.click());
    await page.waitForFunction(
        () => Array.from(document.querySelectorAll('[role="tab"]')).some((tab) => tab.getAttribute("data-key") === "1" && tab.getAttribute("aria-selected") === "true"),
        null, { timeout: 60000 },
    );
    await page.getByText("LIVE CAMERA / WEBRTC", { exact: true }).waitFor({ state: "visible", timeout: 120000 });
    return { text: await page.locator("body").innerText(), snapshot: await ui.snapshot({ full: true }) };
}