import copy
from enum import Enum

from bs4 import BeautifulSoup, Comment, Tag


class NodeLabel(Enum):
    OUTSIDE = "outside"
    INSIDE = "inside"
    BOUNDARY = "boundary"


def extract_sections(sections: list[tuple[Tag, Tag]]) -> Tag:
    """
    Extract multiple HTML sections, replacing out-of-range content with placeholders.

    Args:
        sections: List of (start_element, end_element) tuples defining sections to keep.

    Returns:
        BeautifulSoup Tag object representing the root with placeholders applied.

    Raises:
        ValueError: If elements are invalid or not in the same document.
    """
    if not sections:
        raise ValueError("At least one section must be provided")

    # Step 1: Validate all sections
    _validate_sections(sections)

    # Step 2: Assign document-order indices to all tags
    root = _get_root(sections[0][0])
    tag_order = _build_document_order(root)

    # Step 3: Label every node
    labels = _label_nodes(sections, tag_order, root)

    # Step 4: Clone the tree using labels
    result = _clone_with_labels(root, labels)

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_sections(sections):
    """Validate that all elements are Tags in the same document."""
    root = None
    for start, end in sections:
        if not isinstance(start, Tag) or not isinstance(end, Tag):
            raise ValueError("All start and end elements must be BeautifulSoup Tag objects")

        current_root = _get_root(start)
        if root is None:
            root = current_root
        if _get_root(start) != root or _get_root(end) != root:
            raise ValueError("All elements must be in the same document")


def _get_root(element):
    """Get the root of the document tree."""
    current = element
    while current.parent is not None:
        current = current.parent
    return current


# ---------------------------------------------------------------------------
# Document order indexing
# ---------------------------------------------------------------------------


def _build_document_order(root):
    """
    Walk the tree in document order and assign an index to each Tag.
    Returns a dict mapping Tag -> int index.
    """
    order = {}
    index = 0

    def walk(node):
        nonlocal index
        if isinstance(node, Tag):
            order[node] = index
            index += 1
            for child in node.children:
                walk(child)

    walk(root)
    return order


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------


def _label_nodes(sections, tag_order, root):
    """
    Label every Tag node as INSIDE, BOUNDARY, or OUTSIDE.

    1. For each section (start, end), mark all tags with document-order index
       between start and end (inclusive) as INSIDE.
    2. Walk up from every INSIDE node to root, marking ancestors as BOUNDARY
       (unless already INSIDE).
    3. Everything else stays OUTSIDE.
    """
    labels = {}

    # Collect all inside ranges
    for start, end in sections:
        start_idx = tag_order[start]
        end_idx = tag_order[end]

        # Ensure start comes before end in document order
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        for tag, idx in tag_order.items():
            if start_idx <= idx <= end_idx:
                labels[tag] = NodeLabel.INSIDE

    # Mark boundary nodes (ancestors of INSIDE nodes that aren't themselves INSIDE)
    inside_nodes = [tag for tag, label in labels.items() if label == NodeLabel.INSIDE]
    for tag in inside_nodes:
        current = tag.parent
        while current is not None:
            if current in labels and labels[current] == NodeLabel.INSIDE:
                # Already inside — stop, all ancestors above are also handled
                break
            if current in labels and labels[current] == NodeLabel.BOUNDARY:
                # Already marked — stop, ancestors above are also handled
                break
            labels[current] = NodeLabel.BOUNDARY
            current = current.parent

    return labels


def _get_label(tag, labels):
    """Get the label of a tag, defaulting to OUTSIDE."""
    return labels.get(tag, NodeLabel.OUTSIDE)


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------


def _clone_with_labels(node, labels):
    """
    Recursively clone the tree using labels.

    - INSIDE  -> deep copy (keep everything)
    - BOUNDARY -> clone tag, recurse into children
    - OUTSIDE -> should not be called directly (handled by parent as placeholder)
    """
    label = _get_label(node, labels)

    if label == NodeLabel.INSIDE:
        return copy.deepcopy(node)

    if label == NodeLabel.OUTSIDE:
        # Shouldn't normally reach here for root call, but handle gracefully
        return _create_placeholder_tag(node)

    # BOUNDARY: clone the tag itself, then process children
    soup = BeautifulSoup("", "html.parser")
    new_tag = soup.new_tag(node.name)
    for attr, value in node.attrs.items():
        new_tag[attr] = value

    # Get tag children and group them
    children = [child for child in node.children if isinstance(child, Tag)]
    _process_children(children, labels, new_tag, soup)

    return new_tag


def _process_children(children, labels, parent_tag, soup):
    """
    Process a list of children: keep INSIDE/BOUNDARY, group OUTSIDE into placeholders.
    """
    # Group consecutive OUTSIDE children together
    groups = _group_children_by_label(children, labels)

    for group_type, group_children in groups:
        if group_type == NodeLabel.OUTSIDE:
            # Replace with placeholder pattern
            placeholders = _apply_placeholder_pattern(group_children, soup)
            for p in placeholders:
                parent_tag.append(p)
        else:
            # INSIDE or BOUNDARY — process each individually
            for child in group_children:
                cloned = _clone_with_labels(child, labels)
                parent_tag.append(cloned)


def _group_children_by_label(children, labels):
    """
    Group consecutive children by whether they're OUTSIDE or not.
    Returns list of (label_type, [children]) where label_type is
    OUTSIDE or INSIDE/BOUNDARY (grouped as "kept").
    """
    groups = []
    current_group = []
    current_type = None

    for child in children:
        label = _get_label(child, labels)
        # Treat INSIDE and BOUNDARY the same: "kept"
        is_outside = label == NodeLabel.OUTSIDE

        if current_type is None:
            current_type = NodeLabel.OUTSIDE if is_outside else NodeLabel.INSIDE
            current_group = [child]
        elif is_outside and current_type == NodeLabel.OUTSIDE:
            current_group.append(child)
        elif not is_outside and current_type != NodeLabel.OUTSIDE:
            current_group.append(child)
        else:
            groups.append((current_type, current_group))
            current_type = NodeLabel.OUTSIDE if is_outside else NodeLabel.INSIDE
            current_group = [child]

    if current_group:
        groups.append((current_type, current_group))

    return groups


# ---------------------------------------------------------------------------
# Placeholder helpers (same logic as original)
# ---------------------------------------------------------------------------


def _create_placeholder_tag(original_tag, soup=None):
    """Create an empty tag with '...' content."""
    if soup is None:
        soup = BeautifulSoup("", "html.parser")
    new_tag = soup.new_tag(original_tag.name)
    new_tag.string = "..."
    return new_tag


def _apply_placeholder_pattern(siblings_list, soup):
    """
    Replace a list of siblings with placeholder pattern.

    Pattern:
    - 0 elements: []
    - 1 element:  [<tag>...</tag>]
    - 2 elements: [<tag>...</tag>, <tag>...</tag>]
    - 3+ elements: [<tag>...</tag>, <!-- N-2 omitted -->, <tag>...</tag>]
    """
    n = len(siblings_list)

    if n == 0:
        return []
    elif n == 1:
        return [_create_placeholder_tag(siblings_list[0], soup)]
    elif n == 2:
        return [
            _create_placeholder_tag(siblings_list[0], soup),
            _create_placeholder_tag(siblings_list[1], soup),
        ]
    else:
        first = _create_placeholder_tag(siblings_list[0], soup)
        comment = Comment(f" {n - 2} elements omitted ")
        last = _create_placeholder_tag(siblings_list[-1], soup)
        return [first, comment, last]


# ---------------------------------------------------------------------------
# Example / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    html_content = """
    <html>
    <body>
    <div class="container">
      <p>sibling0</p>
      <div id="ancestor">
        <p>sibling1</p>
        <p>sibling2</p>
        <p>sibling3</p>
        <p>sibling4</p>
        <div>
          <span>before start 1</span>
          <span>before start 2</span>
          <span>before start 3</span>
          <span id="start1">Start 1</span>
          <span>middle</span>
        </div>
        <div>
          <span>more middle</span>
        </div>
        <form id="end1">End 1</form>
        <p>gap between sections</p>
        <p>gap between sections</p>
        <p>gap between sections</p>
        <p>gap between sections</p>
        <p>gap between sections</p>
        <p>gap between sections</p>
        
        <div id="start2">
          <span>Start 2 content</span>
        </div>
        <p>sibling5</p>
        <div id="end2">
          <span>End 2 content</span>
        </div>
        <p>sibling6</p>
        <p>sibling7</p>
      </div>
    </div>
    </body>
    </html>
    """

    soup = BeautifulSoup(html_content, "html.parser")
    start1 = soup.find("span", id="start1")
    end1 = soup.find("form", id="end1")
    start2 = soup.find("div", id="start2")
    end2 = soup.find("div", id="end2")

    # Single section (backward compatible)
    result = extract_sections([(start1, end1)])
    print("=== Single section ===")
    print(result.prettify())

    # Multiple sections
    result = extract_sections([(start1, end1), (start2, end2)])
    print("=== Multiple sections ===")
    print(result.prettify())
