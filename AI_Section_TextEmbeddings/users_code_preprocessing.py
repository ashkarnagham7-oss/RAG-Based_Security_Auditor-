from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pathspec import GitIgnoreSpec
from tree_sitter import Node, Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language


# ============================================================
# OpenAI configuration
# ============================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY was not found. "
        "Add it to the .env file in your project directory."
    )

OPENAI_MODEL = "gpt-5-mini"
MAX_DESCRIPTION_RETRIES = 3
RETRY_DELAY_SECONDS = 3

client = OpenAI(
    timeout=120.0,
    max_retries=0,
)


# ============================================================
# Supported languages
# ============================================================

class LanguageEnum(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


FILE_EXTENSION_LANGUAGE_MAP = {
    ".py": LanguageEnum.PYTHON,
    ".js": LanguageEnum.JAVASCRIPT,
    ".ts": LanguageEnum.TYPESCRIPT,
    ".tsx": LanguageEnum.TYPESCRIPT,
}


# ============================================================
# Tree-sitter definition queries
# ============================================================

LANGUAGE_QUERIES = {
    LanguageEnum.PYTHON: {
        "class_query": """
            (class_definition
                name: (identifier) @class.name)
        """,
        "method_query": """
            (function_definition
                name: (identifier) @function.name)
        """,
    },

    LanguageEnum.JAVASCRIPT: {
        "class_query": """
            (class_declaration
                name: (identifier) @class.name)
        """,
        "method_query": """
            (function_declaration
                name: (identifier) @function.name)

            (method_definition
                name: (property_identifier) @method.name)

            (variable_declarator
                name: (identifier) @function.name
                value: [
                    (arrow_function)
                    (function_expression)
                ])
        """,
    },

    LanguageEnum.TYPESCRIPT: {
        "class_query": """
            (class_declaration
                name: (type_identifier) @class.name)

            (class_declaration
                name: (identifier) @class.name)
        """,
        "method_query": """
            (function_declaration
                name: (identifier) @function.name)

            (method_definition
                name: (property_identifier) @method.name)

            (variable_declarator
                name: (identifier) @function.name
                value: [
                    (arrow_function)
                    (function_expression)
                ])
        """,
    },
}


# ============================================================
# Tree-sitter reference queries
# ============================================================

REFERENCE_QUERIES = {
    LanguageEnum.PYTHON: {
        "class_ref_query": """
            (type (identifier) @class.ref)

            (class_definition
                superclasses: (argument_list
                    (identifier) @class.ref))
        """,
        "method_ref_query": """
            (call
                function: (identifier) @method.ref)

            (call
                function: (attribute
                    attribute: (identifier) @method.ref))
        """,
    },

    LanguageEnum.JAVASCRIPT: {
        "class_ref_query": """
            (new_expression
                constructor: (identifier) @class.ref)

            (class_heritage
                (identifier) @class.ref)
        """,
        "method_ref_query": """
            (call_expression
                function: (identifier) @method.ref)

            (call_expression
                function: (member_expression
                    property: (property_identifier) @method.ref))
        """,
    },

    LanguageEnum.TYPESCRIPT: {
        "class_ref_query": """
            (new_expression
                constructor: (identifier) @class.ref)

            (class_heritage
                (identifier) @class.ref)
        """,
        "method_ref_query": """
            (call_expression
                function: (identifier) @method.ref)

            (call_expression
                function: (member_expression
                    property: (property_identifier) @method.ref))
        """,
    },
}


# ============================================================
# File filtering
# ============================================================

IGNORED_DIRECTORIES = [
    ".git/",
    ".github/",
    ".idea/",
    ".vscode/",
    ".DS_Store",
    "thumbs.db",
    "node_modules/",
    "vendor/",
    "packages/",
    "Go/pkg/",
    ".cargo/",
    "bin/",
    "obj/",
    "dist/",
    "build/",
    "out/",
    "target/",
    "pkg/",
    "release/",
    "debug/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    "venv/",
    ".venv/",
    "env/",
    ".env_env/",
    ".bundle/",
    ".sass-cache/",
    "public/assets/",
    "storage/framework/",
    "storage/logs/",
    ".next/",
    ".nuxt/",
    ".svelte-kit/",
    ".docusaurus/",
    "artifacts/",
    "AppPackages/",
]

VALID_EXTENSIONS = [
    ".py",
    ".js",
    ".ts",
    ".tsx",
]


# ============================================================
# Data objects
# ============================================================

@dataclass
class MethodNode:
    name: str
    method_code: str
    node: Node
    class_name: str | None
    start_line: int
    end_line: int
    unit_type: str = "function"


@dataclass
class ClassNode:
    name: str
    method_declarations: list[str]
    node: Node
    class_code: str
    start_line: int
    end_line: int


# ============================================================
# Functional-description generation
# ============================================================

FUNCTION_DESCRIPTION_INSTRUCTIONS = """
You generate abstract functional descriptions of source-code functions.

Describe only what the supplied function does.

Requirements:
- Describe its purpose and main behaviour.
- Mention important inputs, operations, and outputs.
- Use implementation-independent language where possible.
- Do not discuss vulnerabilities, weaknesses, attacks, risks, security,
  remediation, fixes, secure coding, CWE, CVE, OWASP, or severity.
- Do not judge whether the code is safe or unsafe.
- Do not compare it with another implementation.
- Do not reproduce the source code.
- Do not use Markdown or code fences.
- Return exactly one concise sentence.
""".strip()


def get_function_description(
    source_code: str,
    language: str,
    function_name: str,
) -> tuple[str | None, str | None]:
    """
    Generate one functional description.

    Returns:
        (description, None) when successful.
        (None, error_message) when all attempts fail.
    """

    user_input = f"""
Programming language: {language}
Function name: {function_name}

Source code:
{source_code}
""".strip()

    last_error: Exception | None = None

    for attempt in range(1, MAX_DESCRIPTION_RETRIES + 1):
        try:
            response = client.responses.create(
                model=OPENAI_MODEL,
                instructions=FUNCTION_DESCRIPTION_INSTRUCTIONS,
                reasoning={
                    "effort": "low"
                },
                input=user_input,
                max_output_tokens=300,
            )

            description = response.output_text.strip()

            if not description:
                print("\nEMPTY DESCRIPTION DEBUG")
                print(f"Function: {function_name}")
                print(f"Response ID: {response.id}")
                print(f"Status: {response.status}")
                print(
                        "Incomplete details:",
                        response.incomplete_details,
                    )
                print("Raw output:")
                print(response.output)

                raise ValueError(
                    "GPT-5 mini returned an empty description. "
                    f"status={response.status}, "
                    f"incomplete_details={response.incomplete_details}"
                )

            # Store the description as one clean CSV line.
            description = " ".join(description.split())

            return description, None

        except Exception as error:
            last_error = error

            print(
                f"Description attempt "
                f"{attempt}/{MAX_DESCRIPTION_RETRIES} failed "
                f"for '{function_name}': {error}"
            )

            if attempt < MAX_DESCRIPTION_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

    return None, str(last_error)


# ============================================================
# File discovery
# ============================================================

def get_language_from_extension(
    file_extension: str,
) -> LanguageEnum:
    return FILE_EXTENSION_LANGUAGE_MAP.get(
        file_extension.lower(),
        LanguageEnum.UNKNOWN,
    )


def get_ignore_spec(root: Path) -> GitIgnoreSpec:
    patterns = IGNORED_DIRECTORIES.copy()
    ignore_file = root / ".gitignore"

    if ignore_file.exists():
        with ignore_file.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            for line in file:
                stripped_line = line.strip()

                if stripped_line and not stripped_line.startswith("#"):
                    patterns.append(stripped_line)

    return GitIgnoreSpec.from_lines(patterns)


def get_valid_files(
    root: Path,
    spec: GitIgnoreSpec,
) -> list[tuple[str, LanguageEnum]]:
    valid_files: list[tuple[str, LanguageEnum]] = []

    for base_directory, sub_directories, file_names in os.walk(root):
        # Prevent os.walk from entering ignored folders.
        for sub_directory in sub_directories[:]:
            full_directory = Path(base_directory) / sub_directory
            relative_directory = full_directory.relative_to(root)
            normalized_directory = (
                relative_directory.as_posix() + "/"
            )

            if spec.match_file(normalized_directory):
                sub_directories.remove(sub_directory)

        for file_name in file_names:
            absolute_path = Path(base_directory) / file_name
            relative_path = absolute_path.relative_to(root)
            normalized_relative_path = relative_path.as_posix()

            if spec.match_file(normalized_relative_path):
                continue

            matched_extension = next(
                (
                    extension
                    for extension in VALID_EXTENSIONS
                    if normalized_relative_path.endswith(extension)
                ),
                None,
            )

            if matched_extension is None:
                continue

            language = get_language_from_extension(
                matched_extension
            )

            if language == LanguageEnum.UNKNOWN:
                continue

            normalized_absolute_path = (
                absolute_path.resolve().as_posix()
            )

            valid_files.append(
                (
                    normalized_absolute_path,
                    language,
                )
            )

            print(
                f"Accepted: {normalized_relative_path} "
                f"({language.value})"
            )

    return valid_files


# ============================================================
# Tree-sitter helpers
# ============================================================

def node_text(node: Node) -> str:
    return node.text.decode(
        "utf-8",
        errors="replace",
    )


def is_descendant_of(
    node: Node,
    ancestor: Node,
) -> bool:
    current = node.parent

    while current is not None:
        if current == ancestor:
            return True

        current = current.parent

    return False


def find_parent_class_name(
    node: Node,
    class_nodes: list[Node],
    class_name_by_node_id: dict[int, str],
) -> str | None:
    for class_node in class_nodes:
        if is_descendant_of(node, class_node):
            return class_name_by_node_id.get(class_node.id)

    return None


def get_definition_node(name_node: Node) -> Node:
    """
    Return the whole function/method declaration associated with
    a captured name node.
    """

    definition_node = name_node.parent

    # Include Python decorators when a function is decorated.
    if (
        definition_node is not None
        and definition_node.parent is not None
        and definition_node.parent.type == "decorated_definition"
    ):
        return definition_node.parent

    return definition_node


# ============================================================
# Class method declarations
# ============================================================

def extract_class_methods(
    class_node: Node,
    language_enum: LanguageEnum,
) -> list[str]:
    language = get_language(language_enum.value)

    query = Query(
        language,
        LANGUAGE_QUERIES[language_enum]["method_query"],
    )

    captures = QueryCursor(query).captures(class_node)

    declarations: list[str] = []
    seen_locations: set[tuple[int, int]] = set()

    for capture_name, nodes in captures.items():
        if capture_name not in {
            "method.name",
            "function.name",
        }:
            continue

        for name_node in nodes:
            definition_node = get_definition_node(name_node)

            location = (
                definition_node.start_byte,
                definition_node.end_byte,
            )

            if location in seen_locations:
                continue

            seen_locations.add(location)
            declarations.append(node_text(definition_node))

    return declarations


# ============================================================
# JavaScript/TypeScript route handlers
# ============================================================

HTTP_ROUTE_METHODS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "options",
    "head",
    "all",
    "use",
}


def clean_route_path(path_text: str) -> str:
    path_text = path_text.strip()

    if (
        len(path_text) >= 2
        and path_text[0] in {"'", '"', "`"}
        and path_text[-1] == path_text[0]
    ):
        return path_text[1:-1]

    return path_text


def walk_tree(node: Node):
    """Yield every node in a Tree-sitter tree."""

    yield node

    for child in node.children:
        yield from walk_tree(child)


def extract_route_handlers(
    root_node: Node,
    class_nodes: list[Node],
    class_name_by_node_id: dict[int, str],
) -> list[MethodNode]:
    """
    Extract callbacks such as:

        app.get("/users", async (req, res) => {...})
        router.post("/login", function (req, res) {...})

    The route callback is treated as a function-like unit and
    stored in method_data.csv.
    """

    route_results: list[MethodNode] = []

    for node in walk_tree(root_node):
        if node.type != "call_expression":
            continue

        function_node = node.child_by_field_name("function")
        arguments_node = node.child_by_field_name("arguments")

        if (
            function_node is None
            or arguments_node is None
            or function_node.type != "member_expression"
        ):
            continue

        method_node = function_node.child_by_field_name(
            "property"
        )

        if method_node is None:
            continue

        route_method = node_text(method_node).lower()

        if route_method not in HTTP_ROUTE_METHODS:
            continue

        named_arguments = arguments_node.named_children

        if not named_arguments:
            continue

        route_path_node: Node | None = None
        handler_node: Node | None = None

        for argument in named_arguments:
            if (
                route_path_node is None
                and argument.type in {
                    "string",
                    "template_string",
                }
            ):
                route_path_node = argument

            if argument.type in {
                "arrow_function",
                "function_expression",
            }:
                handler_node = argument

        if route_path_node is None or handler_node is None:
            continue

        route_path = clean_route_path(
            node_text(route_path_node)
        )

        synthetic_name = (
            f"{route_method.upper()} {route_path}"
        )

        parent_class_name = find_parent_class_name(
            node,
            class_nodes,
            class_name_by_node_id,
        )

        route_results.append(
            MethodNode(
                name=synthetic_name,
                method_code=node_text(node),
                node=node,
                class_name=parent_class_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                unit_type="route_handler",
            )
        )

    return route_results


# ============================================================
# File parsing
# ============================================================

def parse_and_extract(
    file_bytes: bytes,
    language_enum: LanguageEnum,
) -> tuple[list[ClassNode], list[MethodNode]]:
    language = get_language(language_enum.value)
    parser = Parser(language)

    tree = parser.parse(file_bytes)
    root_node = tree.root_node

    class_results: list[ClassNode] = []
    method_results: list[MethodNode] = []

    class_nodes: list[Node] = []
    class_name_by_node_id: dict[int, str] = {}

    # --------------------------------------------------------
    # Extract classes
    # --------------------------------------------------------

    class_query = Query(
        language,
        LANGUAGE_QUERIES[language_enum]["class_query"],
    )

    class_captures = QueryCursor(
        class_query
    ).captures(root_node)

    seen_classes: set[tuple[int, int]] = set()

    for capture_name, nodes in class_captures.items():
        if capture_name != "class.name":
            continue

        for name_node in nodes:
            class_node = name_node.parent

            if class_node is None:
                continue

            location = (
                class_node.start_byte,
                class_node.end_byte,
            )

            if location in seen_classes:
                continue

            seen_classes.add(location)

            class_name = node_text(name_node)

            class_nodes.append(class_node)
            class_name_by_node_id[class_node.id] = class_name

            method_declarations = extract_class_methods(
                class_node,
                language_enum,
            )

            class_results.append(
                ClassNode(
                    name=class_name,
                    method_declarations=method_declarations,
                    node=class_node,
                    class_code=node_text(class_node),
                    start_line=class_node.start_point[0] + 1,
                    end_line=class_node.end_point[0] + 1,
                )
            )

    # --------------------------------------------------------
    # Extract named functions and methods
    # --------------------------------------------------------

    method_query = Query(
        language,
        LANGUAGE_QUERIES[language_enum]["method_query"],
    )

    method_captures = QueryCursor(
        method_query
    ).captures(root_node)

    seen_methods: set[tuple[int, int]] = set()

    for capture_name, nodes in method_captures.items():
        if capture_name not in {
            "method.name",
            "function.name",
        }:
            continue

        for name_node in nodes:
            definition_node = get_definition_node(name_node)

            if definition_node is None:
                continue

            location = (
                definition_node.start_byte,
                definition_node.end_byte,
            )

            if location in seen_methods:
                continue

            seen_methods.add(location)

            method_name = node_text(name_node)

            parent_class_name = find_parent_class_name(
                definition_node,
                class_nodes,
                class_name_by_node_id,
            )

            unit_type = (
                "class_method"
                if parent_class_name is not None
                else "function"
            )

            method_results.append(
                MethodNode(
                    name=method_name,
                    method_code=node_text(definition_node),
                    node=definition_node,
                    class_name=parent_class_name,
                    start_line=(
                        definition_node.start_point[0] + 1
                    ),
                    end_line=(
                        definition_node.end_point[0] + 1
                    ),
                    unit_type=unit_type,
                )
            )

    # --------------------------------------------------------
    # Extract anonymous JS/TS route callbacks
    # --------------------------------------------------------

    if language_enum in {
        LanguageEnum.JAVASCRIPT,
        LanguageEnum.TYPESCRIPT,
    }:
        route_handlers = extract_route_handlers(
            root_node,
            class_nodes,
            class_name_by_node_id,
        )

        for route_handler in route_handlers:
            location = (
                route_handler.node.start_byte,
                route_handler.node.end_byte,
            )

            if location not in seen_methods:
                seen_methods.add(location)
                method_results.append(route_handler)

    return class_results, method_results


# ============================================================
# Repository parsing and description generation
# ============================================================

def build_method_id(
    codebase_root: Path,
    file_path: str,
    method_node: MethodNode,
) -> str:
    relative_path = Path(file_path).resolve().relative_to(
        codebase_root.resolve()
    )

    return (
        f"{relative_path.as_posix()}:"
        f"{method_node.start_line}:"
        f"{method_node.end_line}:"
        f"{method_node.name}"
    )


def parse_files(
    valid_files: list[tuple[str, LanguageEnum]],
    codebase_root: Path,
) -> tuple[
    list[dict],
    list[dict],
    set[str],
    set[str],
    list[dict],
]:
    class_data: list[dict] = []
    method_data: list[dict] = []
    description_errors: list[dict] = []

    all_class_names: set[str] = set()
    all_method_names: set[str] = set()

    total_files = len(valid_files)

    for file_position, (file_path, language) in enumerate(
        valid_files,
        start=1,
    ):
        print(
            f"\nParsing file {file_position}/{total_files}: "
            f"{file_path}"
        )

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="replace",
            ) as file:
                source_code = file.read()

            file_bytes = source_code.encode("utf-8")

            class_nodes, method_nodes = parse_and_extract(
                file_bytes,
                language,
            )

        except Exception as error:
            print(f"Failed to parse {file_path}: {error}")
            continue

        relative_file_path = (
            Path(file_path)
            .resolve()
            .relative_to(codebase_root.resolve())
            .as_posix()
        )

        # ----------------------------------------------------
        # Store class records
        # ----------------------------------------------------

        for class_node in class_nodes:
            all_class_names.add(class_node.name)

            class_data.append({
                "file_path": relative_file_path,
                "language": language.value,
                "class_name": class_node.name,
                "start_line": class_node.start_line,
                "end_line": class_node.end_line,
                "method_declarations": (
                    "\n--------\n".join(
                        class_node.method_declarations
                    )
                    if class_node.method_declarations
                    else ""
                ),
                "source_code": class_node.class_code,
                "references": [],
            })

        # ----------------------------------------------------
        # Generate function descriptions
        # ----------------------------------------------------

        for method_position, method_node in enumerate(
            method_nodes,
            start=1,
        ):
            all_method_names.add(method_node.name)

            method_id = build_method_id(
                codebase_root,
                file_path,
                method_node,
            )

            print(
                f"  Describing function "
                f"{method_position}/{len(method_nodes)}: "
                f"{method_node.name}"
            )

            description, error_message = (
                get_function_description(
                    source_code=method_node.method_code,
                    language=language.value,
                    function_name=method_node.name,
                )
            )

            if description is None:
                description = ""

                description_errors.append({
                    "method_id": method_id,
                    "file_path": relative_file_path,
                    "language": language.value,
                    "function_name": method_node.name,
                    "start_line": method_node.start_line,
                    "end_line": method_node.end_line,
                    "error": error_message or "Unknown error",
                })

                print(
                    f"  Description failed for "
                    f"{method_node.name}; processing continues."
                )

            method_data.append({
                "method_id": method_id,
                "unit_type": method_node.unit_type,
                "file_path": relative_file_path,
                "language": language.value,
                "class_name": (
                    method_node.class_name or ""
                ),
                "name": method_node.name,
                "start_line": method_node.start_line,
                "end_line": method_node.end_line,
                "source_code": method_node.method_code,
                "functional_description": description,
                "references": [],
            })

    return (
        class_data,
        method_data,
        all_class_names,
        all_method_names,
        description_errors,
    )


# ============================================================
# Reference extraction
# ============================================================

def find_references(
    file_list: list[tuple[str, LanguageEnum]],
    class_names: set[str],
    method_names: set[str],
    codebase_root: Path,
) -> dict:
    references = {
        "class": defaultdict(list),
        "method": defaultdict(list),
    }

    files_by_language = defaultdict(list)

    for file_path, language in file_list:
        files_by_language[language].append(file_path)

    for language_enum, files in files_by_language.items():
        configuration = REFERENCE_QUERIES.get(language_enum)

        if configuration is None:
            continue

        language = get_language(language_enum.value)
        parser = Parser(language)

        class_ref_query = Query(
            language,
            configuration["class_ref_query"],
        )

        method_ref_query = Query(
            language,
            configuration["method_ref_query"],
        )

        for file_path in files:
            try:
                with open(
                    file_path,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as file:
                    source_code = file.read()

                tree = parser.parse(
                    source_code.encode("utf-8")
                )

            except Exception as error:
                print(
                    f"Reference parsing failed for "
                    f"{file_path}: {error}"
                )
                continue

            root_node = tree.root_node

            relative_file_path = (
                Path(file_path)
                .resolve()
                .relative_to(codebase_root.resolve())
                .as_posix()
            )

            # Class references
            class_captures = QueryCursor(
                class_ref_query
            ).captures(root_node)

            for capture_name, nodes in class_captures.items():
                if capture_name != "class.ref":
                    continue

                for node in nodes:
                    name = node_text(node)

                    if name not in class_names:
                        continue

                    references["class"][name].append({
                        "file": relative_file_path,
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1] + 1,
                        "text": (
                            node_text(node.parent)
                            if node.parent is not None
                            else name
                        ),
                    })

            # Function and method references
            method_captures = QueryCursor(
                method_ref_query
            ).captures(root_node)

            for capture_name, nodes in method_captures.items():
                if capture_name != "method.ref":
                    continue

                for node in nodes:
                    name = node_text(node)

                    if name not in method_names:
                        continue

                    references["method"][name].append({
                        "file": relative_file_path,
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1] + 1,
                        "text": (
                            node_text(node.parent)
                            if node.parent is not None
                            else name
                        ),
                    })

    return references


def attach_references(
    class_data: list[dict],
    method_data: list[dict],
    references: dict,
) -> None:
    """
    Attach references without converting records into dictionaries
    keyed only by class or method name.

    This prevents duplicate names from deleting records.
    """

    for class_record in class_data:
        class_name = class_record["class_name"]

        class_record["references"] = (
            references["class"].get(
                class_name,
                [],
            )
        )

    for method_record in method_data:
        method_name = method_record["name"]

        if method_record["unit_type"] == "route_handler":
            method_record["references"] = []
        else:
            method_record["references"] = (
                references["method"].get(
                    method_name,
                    [],
                )
            )


# ============================================================
# CSV output
# ============================================================

def create_output_directory(
    codebase_path: Path,
) -> Path:
    codebase_folder_name = codebase_path.resolve().name

    output_directory = (
        Path("processed") / codebase_folder_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def serialize_references(
    references: list[dict],
) -> str:
    return "; ".join(
        (
            f"{reference['file']}:"
            f"{reference['line']}:"
            f"{reference['column']}"
        )
        for reference in references
    )


def write_class_data_to_csv(
    class_data: list[dict],
    output_directory: Path,
) -> None:
    output_file = output_directory / "class_data.csv"

    fieldnames = [
        "file_path",
        "language",
        "class_name",
        "start_line",
        "end_line",
        "method_declarations",
        "source_code",
        "references",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in class_data:
            output_record = dict(record)
            output_record["references"] = serialize_references(
                record.get("references", [])
            )

            writer.writerow(output_record)

    print(f"Class data written to: {output_file.resolve()}")


def write_method_data_to_csv(
    method_data: list[dict],
    output_directory: Path,
) -> None:
    output_file = output_directory / "method_data.csv"

    fieldnames = [
        "method_id",
        "unit_type",
        "file_path",
        "language",
        "class_name",
        "name",
        "start_line",
        "end_line",
        "source_code",
        "functional_description",
        "references",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in method_data:
            output_record = dict(record)
            output_record["references"] = serialize_references(
                record.get("references", [])
            )

            writer.writerow(output_record)

    print(f"Method data written to: {output_file.resolve()}")


def write_description_errors_to_csv(
    description_errors: list[dict],
    output_directory: Path,
) -> None:
    output_file = (
        output_directory / "description_errors.csv"
    )

    fieldnames = [
        "method_id",
        "file_path",
        "language",
        "function_name",
        "start_line",
        "end_line",
        "error",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(description_errors)

    print(
        f"Description errors written to: "
        f"{output_file.resolve()}"
    )


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
 
    codebase_path = Path(
        r"C:\Users\ASUST\Desktop\RAG Based Security Auditor\BenchmarkPython-main\testcode"
    )

    if not codebase_path.exists():
        raise FileNotFoundError(
            f"Codebase directory does not exist: "
            f"{codebase_path.resolve()}"
        )

    print("=" * 70)
    print("SOURCE-CODE PREPROCESSING")
    print("=" * 70)
    print(f"Codebase: {codebase_path.resolve()}")
    print(f"Description model: {OPENAI_MODEL}")

    ignore_spec = get_ignore_spec(codebase_path)

    valid_files = get_valid_files(
        codebase_path,
        ignore_spec,
    )

    print(f"\nAccepted source files: {len(valid_files)}")

    (
        class_data,
        method_data,
        class_names,
        method_names,
        description_errors,
    ) = parse_files(
        valid_files,
        codebase_path,
    )

    print("\nFinding references...")

    references = find_references(
        valid_files,
        class_names,
        method_names,
        codebase_path,
    )

    attach_references(
        class_data,
        method_data,
        references,
    )

    output_directory = create_output_directory(
        codebase_path
    )

    write_class_data_to_csv(
        class_data,
        output_directory,
    )

    write_method_data_to_csv(
        method_data,
        output_directory,
    )

    write_description_errors_to_csv(
        description_errors,
        output_directory,
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)
    print(f"Classes extracted: {len(class_data)}")
    print(f"Functions extracted: {len(method_data)}")
    print(
        f"Descriptions generated: "
        f"{len(method_data) - len(description_errors)}"
    )
    print(
        f"Description failures: "
        f"{len(description_errors)}"
    )
    print(
        f"Output directory: "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
