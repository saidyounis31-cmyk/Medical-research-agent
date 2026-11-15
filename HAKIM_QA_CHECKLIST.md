# HAKIM — pre-deployment review checklist

**Status:** Draft; not approved for deployment. This document describes checks to perform, not tests already passed.

## Scope
- The public website on Render must remain unchanged until explicit deployment approval.
- Preserve existing Europe PMC search, request builder, clinical prompt builder, imaging report prompt builder, and lecture prompt builder.
- HAKIM is the sole product brand; credit is “Created by ANEMONE SLN LTD”.
- Current localization covers the landing page, many labels, selected options, and selected status labels. Dynamic result text and some descriptions still require translation.

## Browser smoke tests (Android Chrome and desktop)
- [ ] Load index.html without JavaScript errors; verify the five cards appear.
- [ ] Tap each card; verify it scrolls to the correct existing section.
- [ ] Switch ar → en → he → de → ar; verify direction, headings, labels, selected option values and controls.
- [ ] Verify each dropdown selection remains logically unchanged after language switching.
- [ ] Search Europe PMC for “Nigella sativa” and verify article titles and original-source links; test empty search, open-access filter, and clear.
- [ ] Build a request, copy text, download TXT and print; check generated output and non-empty fields.
- [ ] Build clinical, imaging and lecture prompts with sample text; verify no automated diagnosis or image-reading claim.
- [ ] Verify mobile width 360–430px, keyboard focus, text contrast and no horizontal overflow.
- [ ] Verify no analytics or account functionality is falsely advertised.
- [ ] Verify English/Hebrew/German dynamic messages are translated before claiming full-language support.

## Release gates
1. All functional smoke tests pass.
2. User sees a real browser preview of the development branch.
3. User explicitly approves merge and Render deployment.
