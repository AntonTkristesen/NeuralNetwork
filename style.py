"""Colours, fonts and the bit of CSS the theme in .streamlit/config.toml can't do.

The page is black on white; colour is only used for data, so wherever it appears it means something.
"""

INK = "#15171C"  # text
MUTED = "#69707D"  # secondary text and axis labels
HAIRLINE = "#E3E7EE"  # gridlines
TRACK = "#EEF1F5"  # empty probability bars, the lightest cells of the confusion matrix
NEURON_OFF = "#DDE3EB"  # a neuron that doesn't fire
BLUE = "#2A78D6"  # a connection that pushes the next neuron UP; the validation curve
DEEP_BLUE = "#1C5CAB"  # a neuron firing at full strength
ORANGE = "#EB6834"  # a connection that pushes the next neuron DOWN
GRAY = "#B4BAC4"  # the training curve; probabilities that aren't the guess
GREEN = "#0CA30C"  # the guess is right
GREEN_TEXT = "#0A7D0A"  # the same green, dark enough for text
RED = "#D03B3B"  # the guess is wrong
FONT = "Geist, system-ui, sans-serif"  # loaded by the theme
MONO = "'Geist Mono', ui-monospace, monospace"

# A narrower page, and the guess/actual row in the Test tab
PAGE_CSS = f"""<style>
[data-testid="stMainBlockContainer"] {{ max-width: 1100px; padding-top: 3rem; padding-bottom: 5rem; }}
h1 {{ letter-spacing: -0.03em; }}
.verdict {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-bottom: 0.5rem; }}
.verdict .label {{ font-size: 0.875rem; color: {MUTED}; }}
.verdict .value {{ font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; }}
.verdict .good {{ color: {GREEN_TEXT}; }}
.verdict .bad {{ color: {RED}; }}
@media (max-width: 640px) {{ .verdict {{ grid-template-columns: 1fr; gap: 0.5rem; }} }}
</style>"""
