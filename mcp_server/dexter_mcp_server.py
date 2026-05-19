"""FastMCP server — filesystem, document, and Outlook tools for Dexter."""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import time
from pathlib import Path

from fastmcp import FastMCP

logger = logging.getLogger("dexter_mcp_server")

mcp = FastMCP("dexter-mcp-server")

ALLOWED_ROOTS: list[str] = json.loads(os.environ.get("DEXTER_ALLOWED_ROOTS", "[]"))


def _validate_path(path: str) -> Path:
    """
    Resolve the path and verify it is inside one of the allowed roots.
    Raises PermissionError if outside allowed roots.
    """
    resolved = Path(path).resolve()

    if not ALLOWED_ROOTS:
        raise PermissionError(
            "No allowed roots configured. Cannot access filesystem."
        )

    for root in ALLOWED_ROOTS:
        root_resolved = Path(root).resolve()
        try:
            resolved.relative_to(root_resolved)
            return resolved
        except ValueError:
            continue

    raise PermissionError(
        f"Path '{path}' is outside all allowed roots. Allowed: {ALLOWED_ROOTS}"
    )


@mcp.tool()
async def read_file(path: str) -> dict:
    """Read the complete text content of a file within allowed roots (max 10MB)."""
    try:
        validated = _validate_path(path)

        size = validated.stat().st_size
        if size > 10 * 1024 * 1024:
            return {
                "success": False,
                "error": f"File too large: {size} bytes. Maximum is 10MB.",
            }

        content = None
        encoding_used = None
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                content = validated.read_text(encoding=enc)
                encoding_used = enc
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            return {
                "success": False,
                "error": "File appears to be binary. Cannot read as text.",
            }

        return {
            "success": True,
            "content": content,
            "encoding": encoding_used,
            "size_bytes": size,
            "lines": content.count("\n") + 1,
            "path": str(validated),
        }

    except PermissionError as e:
        return {"success": False, "error": str(e)}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {path}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def write_file(path: str, content: str, mode: str = "overwrite") -> dict:
    """Write text content to a file (overwrite or append)."""
    try:
        if mode not in ("overwrite", "append"):
            return {
                "success": False,
                "error": f"Invalid mode '{mode}'. Must be overwrite or append.",
            }

        validated = _validate_path(path)
        validated.parent.mkdir(parents=True, exist_ok=True)

        if mode == "overwrite":
            validated.write_text(content, encoding="utf-8")
        else:
            with open(validated, "a", encoding="utf-8") as handle:
                handle.write(content)

        bytes_written = len(content.encode("utf-8"))

        return {
            "success": True,
            "bytes_written": bytes_written,
            "path": str(validated),
            "mode": mode,
        }

    except PermissionError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Write failed: {str(e)}"}


@mcp.tool()
async def list_directory(
    path: str,
    pattern: str = "*",
    include_hidden: bool = False,
) -> dict:
    """List files and folders at the given path (up to 500 items)."""
    try:
        validated = _validate_path(path)

        if not validated.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}

        items = []
        for item in validated.glob(pattern):
            if not include_hidden and item.name.startswith("."):
                continue

            try:
                stat = item.stat()
                items.append(
                    {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size_bytes": stat.st_size if item.is_file() else 0,
                        "modified_iso": datetime.datetime.fromtimestamp(
                            stat.st_mtime
                        ).isoformat(),
                        "extension": item.suffix.lower() if item.is_file() else "",
                    }
                )
            except (PermissionError, OSError):
                continue

            if len(items) >= 500:
                return {
                    "success": True,
                    "items": items,
                    "total_count": len(items),
                    "truncated": True,
                    "message": "Results truncated at 500. Use pattern to filter.",
                }

        return {
            "success": True,
            "items": sorted(
                items,
                key=lambda x: (x["type"] == "file", x["name"].lower()),
            ),
            "total_count": len(items),
            "truncated": False,
        }

    except PermissionError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"List failed: {str(e)}"}


@mcp.tool()
async def search_files(
    root: str,
    query: str,
    file_types: list | None = None,
    search_content: bool = False,
    max_results: int = 50,
) -> dict:
    """Search for files by name and optionally content under root."""
    start = time.time()
    deadline = start + 15.0

    try:
        validated_root = _validate_path(root)
        max_results = min(max_results, 100)

        query_lower = query.lower()
        matches = []

        extensions = None
        if file_types:
            extensions = {f".{ext.lstrip('.')}" for ext in file_types}

        for file_path in validated_root.rglob("*"):
            if time.time() > deadline:
                break

            if not file_path.is_file():
                continue

            if extensions and file_path.suffix.lower() not in extensions:
                continue

            if any(part.startswith(".") for part in file_path.parts):
                continue

            if any(
                d in file_path.parts
                for d in (
                    "node_modules",
                    "__pycache__",
                    ".venv",
                    "venv",
                    ".git",
                    "build",
                    "dist",
                    "target",
                )
            ):
                continue

            if query_lower in file_path.name.lower():
                matches.append(
                    {
                        "path": str(file_path),
                        "name": file_path.name,
                        "match_type": "filename",
                        "snippet": "",
                    }
                )
                if len(matches) >= max_results:
                    break
                continue

            if search_content and file_path.stat().st_size < 1024 * 1024:
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    idx = text.lower().find(query_lower)
                    if idx >= 0:
                        start_idx = max(0, idx - 100)
                        end_idx = min(len(text), idx + 200)
                        snippet = text[start_idx:end_idx]
                        snippet = snippet.replace("\n", " ").strip()

                        matches.append(
                            {
                                "path": str(file_path),
                                "name": file_path.name,
                                "match_type": "content",
                                "snippet": snippet,
                            }
                        )

                        if len(matches) >= max_results:
                            break
                except Exception:
                    continue

        elapsed = (time.time() - start) * 1000

        return {
            "success": True,
            "matches": matches,
            "total_matches": len(matches),
            "search_time_ms": round(elapsed, 1),
            "truncated": len(matches) >= max_results,
        }

    except PermissionError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Search failed: {str(e)}"}


@mcp.tool()
async def create_directory(path: str) -> dict:
    """Create a directory and all parent directories."""
    try:
        validated = _validate_path(path)
        already_existed = validated.exists()
        validated.mkdir(parents=True, exist_ok=True)

        return {
            "success": True,
            "path": str(validated),
            "already_existed": already_existed,
        }

    except PermissionError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Create failed: {str(e)}"}


@mcp.tool()
async def read_word_doc(path: str) -> dict:
    """Extract text from a Microsoft Word .docx file."""
    try:
        validated = _validate_path(path)

        if not str(validated).lower().endswith(".docx"):
            return {
                "success": False,
                "error": "Only .docx files supported. Not .doc or other formats.",
            }

        if validated.name.startswith("~$"):
            return {
                "success": False,
                "error": "Temp file (still open in Word). Close Word first.",
            }

        from docx import Document

        doc = Document(str(validated))

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        tables = []
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            tables.append(table_data)

        full_text = "\n".join(paragraphs)
        word_count = len(full_text.split())

        title = ""
        try:
            title = doc.core_properties.title or ""
        except Exception:
            pass

        return {
            "success": True,
            "title": title,
            "paragraphs": paragraphs,
            "tables": tables,
            "full_text": full_text,
            "word_count": word_count,
            "page_estimate": max(1, word_count // 250),
        }

    except PermissionError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Could not read Word doc: {str(e)}"}


@mcp.tool()
async def read_excel(path: str, sheet: str | None = None, max_rows: int = 1000) -> dict:
    """Read data from a Microsoft Excel .xlsx file."""
    try:
        validated = _validate_path(path)
        max_rows = min(max_rows, 5000)

        import openpyxl

        wb = openpyxl.load_workbook(str(validated), read_only=True, data_only=True)

        all_sheets = wb.sheetnames

        if sheet:
            if sheet not in all_sheets:
                wb.close()
                return {
                    "success": False,
                    "error": f"Sheet '{sheet}' not found. Available: {all_sheets}",
                }
            ws = wb[sheet]
        else:
            ws = wb.active

        sheet_name = ws.title
        rows = []
        headers = []

        for i, row in enumerate(ws.iter_rows(values_only=True)):
            row_data = ["" if v is None else str(v) for v in row]

            if i == 0:
                headers = row_data
            else:
                rows.append(row_data)

            if len(rows) >= max_rows:
                wb.close()
                return {
                    "success": True,
                    "sheet_name": sheet_name,
                    "all_sheet_names": all_sheets,
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows),
                    "column_count": len(headers),
                    "truncated": True,
                }

        wb.close()

        return {
            "success": True,
            "sheet_name": sheet_name,
            "all_sheet_names": all_sheets,
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "column_count": len(headers),
            "truncated": False,
        }

    except PermissionError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Could not read Excel: {str(e)}"}


@mcp.tool()
async def read_pdf(path: str, max_pages: int = 50) -> dict:
    """Extract text from a PDF file."""
    try:
        validated = _validate_path(path)
        max_pages = min(max_pages, 200)

        import pypdf

        with open(str(validated), "rb") as handle:
            reader = pypdf.PdfReader(handle)

            if reader.is_encrypted:
                return {
                    "success": False,
                    "error": "PDF is encrypted. Cannot read without password.",
                }

            total_pages = len(reader.pages)
            pages_to_read = min(total_pages, max_pages)

            title = ""
            author = ""
            try:
                meta = reader.metadata
                if meta:
                    title = meta.get("/Title", "") or ""
                    author = meta.get("/Author", "") or ""
            except Exception:
                pass

            pages = []
            full_text_parts = []

            for i in range(pages_to_read):
                try:
                    text = reader.pages[i].extract_text()
                    if not text or not text.strip():
                        text = "[scanned page - no text]"
                except Exception:
                    text = "[page extraction failed]"

                pages.append({"page_number": i + 1, "text": text})
                full_text_parts.append(f"--- Page {i + 1} ---\n{text}")

            return {
                "success": True,
                "title": title,
                "author": author,
                "page_count": total_pages,
                "pages_extracted": pages_to_read,
                "pages": pages,
                "full_text": "\n\n".join(full_text_parts),
                "truncated": total_pages > max_pages,
            }

    except PermissionError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Could not read PDF: {str(e)}"}


@mcp.tool()
async def list_emails(
    folder: str = "Inbox",
    count: int = 10,
    unread_only: bool = False,
) -> dict:
    """List recent emails from Outlook."""
    try:
        import pythoncom

        pythoncom.CoInitialize()

        try:
            import win32com.client

            outlook = win32com.client.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")

            try:
                inbox = namespace.GetDefaultFolder(6)
                if folder == "Inbox":
                    target = inbox
                elif folder == "Sent Items":
                    target = namespace.GetDefaultFolder(5)
                elif folder == "Drafts":
                    target = namespace.GetDefaultFolder(16)
                else:
                    target = inbox.Parent.Folders[folder]
            except Exception:
                return {"success": False, "error": f"Folder '{folder}' not found."}

            messages = target.Items
            messages.Sort("[ReceivedTime]", True)

            emails = []
            count = min(count, 50)

            for msg in messages:
                if len(emails) >= count:
                    break

                try:
                    if unread_only and msg.UnRead is False:
                        continue

                    body = msg.Body or ""
                    body_clean = re.sub(r"<[^>]+>", "", body)[:200].strip()

                    emails.append(
                        {
                            "subject": msg.Subject or "",
                            "sender_name": msg.SenderName or "",
                            "sender_email": msg.SenderEmailAddress or "",
                            "received_time": str(msg.ReceivedTime),
                            "is_unread": bool(msg.UnRead),
                            "body_preview": body_clean,
                            "has_attachments": msg.Attachments.Count > 0,
                        }
                    )
                except Exception:
                    continue

            return {
                "success": True,
                "folder": folder,
                "emails": emails,
                "total_returned": len(emails),
            }

        finally:
            pythoncom.CoUninitialize()

    except ImportError:
        return {
            "success": False,
            "error": "pywin32 not installed or not running on Windows.",
        }
    except Exception as e:
        return {"success": False, "error": f"Outlook error: {str(e)}"}


@mcp.tool()
async def create_email_draft(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
) -> dict:
    """Create an email draft in Outlook (never auto-sends)."""
    try:
        import pythoncom

        pythoncom.CoInitialize()

        try:
            import win32com.client

            outlook = win32com.client.Dispatch("Outlook.Application")

            mail = outlook.CreateItem(0)
            mail.To = to
            mail.Subject = subject
            mail.Body = body
            if cc:
                mail.CC = cc

            mail.Display(True)

            return {
                "success": True,
                "action": "draft_created_and_displayed",
                "to": to,
                "subject": subject,
                "message": "Draft opened in Outlook. Review and send manually.",
            }

        finally:
            pythoncom.CoUninitialize()

    except ImportError:
        return {"success": False, "error": "pywin32 not installed."}
    except Exception as e:
        return {"success": False, "error": f"Could not create draft: {str(e)}"}


@mcp.tool()
async def list_calendar_events(days_ahead: int = 7, days_back: int = 0) -> dict:
    """List calendar events from Outlook Calendar."""
    try:
        import pythoncom
        import datetime as dt

        pythoncom.CoInitialize()

        try:
            import win32com.client

            outlook = win32com.client.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            calendar = namespace.GetDefaultFolder(9)

            days_ahead = min(days_ahead, 90)
            days_back = min(days_back, 30)

            now = dt.datetime.now()
            start_date = now - dt.timedelta(days=days_back)
            end_date = now + dt.timedelta(days=days_ahead)

            items = calendar.Items
            items.IncludeRecurrences = True
            items.Sort("[Start]")

            restriction = (
                f"[Start] >= '{start_date.strftime('%m/%d/%Y')}' "
                f"AND [Start] <= '{end_date.strftime('%m/%d/%Y')}'"
            )

            filtered = items.Restrict(restriction)
            events = []

            for event in filtered:
                try:
                    events.append(
                        {
                            "subject": event.Subject or "",
                            "start": str(event.Start),
                            "end": str(event.End),
                            "location": event.Location or "",
                            "organizer": event.Organizer or "",
                            "is_all_day": bool(event.AllDayEvent),
                            "body_preview": (event.Body or "")[:150].strip(),
                        }
                    )
                except Exception:
                    continue

            return {
                "success": True,
                "events": sorted(events, key=lambda e: e["start"]),
                "total_events": len(events),
                "range_start": start_date.date().isoformat(),
                "range_end": end_date.date().isoformat(),
            }

        finally:
            pythoncom.CoUninitialize()

    except ImportError:
        return {"success": False, "error": "pywin32 not installed."}
    except Exception as e:
        return {"success": False, "error": f"Calendar error: {str(e)}"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
