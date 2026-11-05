"""Cytoscape.js graph. Receives only API-approved JSON; labels are drawn on canvas as text."""

from __future__ import annotations

import json
from typing import Any

import streamlit.components.v1 as components

CYTOSCAPE = "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"

_TEMPLATE = """<div id="cy" style="height:420px;border:1px solid #8884"></div>
<pre id="ev" style="white-space:pre-wrap;font:13px sans-serif;margin:8px 0 0"></pre>
<script src="__SRC__"></script>
<script>
const data = __DATA__;
const cy = cytoscape({container: document.getElementById('cy'),
  elements: [
    ...data.nodes.map(n => ({data: {id: n.id, label: n.label}})),
    ...data.edges.map(e => ({data: {id: e.id, source: e.source, target: e.target,
      label: e.predicate + (e.polarity === 'NEGATED' ? ' (negated)' : ''),
      quote: e.evidence ? e.evidence.title + ': ' + e.evidence.quote : ''}}))],
  style: [{selector: 'node', style: {label: 'data(label)', 'background-color': '#4c78a8'}},
    {selector: 'edge', style: {label: 'data(label)', 'curve-style': 'bezier', width: 2,
      'target-arrow-shape': 'triangle', 'font-size': 9, 'line-color': '#999',
      'target-arrow-color': '#999'}}],
  layout: {name: 'cose'}});
cy.on('tap', 'edge', ev => {
  document.getElementById('ev').textContent = ev.target.data('quote');
});
</script>"""


def render(graph: dict[str, Any], height: int = 520) -> None:
    # "<" is escaped so note-derived strings can never close the script element.
    payload = json.dumps(graph, ensure_ascii=False).replace("<", "\\u003c")
    html = _TEMPLATE.replace("__SRC__", CYTOSCAPE).replace("__DATA__", payload)
    components.html(html, height=height)
