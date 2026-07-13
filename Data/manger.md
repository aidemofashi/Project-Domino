Optimized English Prompt:

You will process the provided chat logs and generate a Markdown-formatted output that organizes extracted information into three memory tiers. If no relevant information is found for a tier, omit that entire section (do not output empty headings).

Memory classification:

Long‑term memory – stable personal attributes of the user: real name, online aliases, gender, hobbies, interests, daily habits, favorite things, and any enduring personal traits.

Medium‑term memory – recent plans, current work tasks, ongoing projects, games they are playing, or any near‑future intentions.

Short‑term memory – immediate context from the most recent conversation: what you and the user were just doing, the current topic, specific events mentioned, or any ongoing dialogue details.

Output format (example):
    ```text

    # 2026-7-11-06:00

    # Long‑term memory
    ## [Category/Item 1]
    ## [Category/Item 2]
    ...

    # Medium‑term memory
    ## [Item 1]
    ## [Item 2]
    ...

    # Short‑term memory
    ## [Item 1]
    ## [Item 2]
    ...

    ```
    Important rules:

        Use ## subheadings under each main memory section to list individual items.

        Preserve the timestamp as shown (YYYY-M-D-HH:MM) in the top‑level heading.

        If a tier has no retrievable information, do not include that tier heading at all.

        Only output the Markdown content; no extra commentary or prose.
    