#!/usr/bin/env python3
"""
Honest Broker DICOM Relabeling Script

Reads DICOM files from an input directory, calls a remote MDA-style
Honest Broker service to get de-identified PatientID and PatientName,
replaces both in the DICOM headers, and writes the modified files to
an output directory preserving the directory structure.

Mirrors the logic from the MDA plugin (mdanderson_plugin):
  - Authenticates via STS: POST https://{stsHost}/token
  - Looks up PatientID: GET https://{apiHost}/DeIdentification/lookup?idIn={PatientID}
  - Looks up PatientName: GET https://{apiHost}/DeIdentification/lookup?idIn={PatientName}
  - Replaces both tags with the HB-returned de-identified values

Usage:
    python honest_broker.py <input_dir> <output_dir> [--config config.yaml] [--dry-run]

Examples:
    python honest_broker.py ../data/Elder_subject_florbetapir ./output
    python honest_broker.py ./input ./output --config my_config.yaml --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

import pydicom
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("honest_broker")


class HonestBrokerClient:
    """Client for the remote MDA-style Honest Broker de-identification service.

    Mirrors HonestBrokerServiceImpl.java from mdanderson_plugin:
      - auth() -> POST https://{stsHost}/token with JSON credentials
      - lookup(idIn) -> GET https://{apiHost}/DeIdentification/lookup?idIn={idIn}
      - Returns first result's idOut field
    """

    def __init__(self, config: dict):
        self.sts_host = config["sts_host"]
        self.api_host = config["api_host"]
        self.app_name = config["app_name"]
        self.app_key = config["app_key"]
        self.username = config["username"]
        self.password = config["password"]
        self.timeout = config.get("timeout", 30)
        self.token_cache_minutes = config.get("token_cache_minutes", 50)

        self._token = None
        self._token_expires_at = 0
        self._id_cache = {}  # original_id -> deidentified_id

    def authenticate(self) -> str:
        """Authenticate with STS and return a JWT token. Caches for configured TTL."""
        if self._token and time.time() < self._token_expires_at:
            log.debug("Using cached STS token (expires in %.0f seconds)",
                      self._token_expires_at - time.time())
            return self._token

        sts_url = f"https://{self.sts_host}/token"
        payload = json.dumps({
            "UserName": self.username,
            "AppName": self.app_name,
            "AppKey": self.app_key,
            "Password": self.password,
        }).encode("utf-8")

        log.info("Authenticating with STS at %s", sts_url)
        start = time.time()

        req = urllib.request.Request(
            sts_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                token = resp.read().decode("utf-8").strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            duration = time.time() - start
            log.error("STS authentication FAILED (%.1fs): HTTP %d: %s",
                      duration, e.code, error_body)
            raise RuntimeError(f"STS authentication failed: HTTP {e.code}: {error_body}") from e
        except Exception as e:
            duration = time.time() - start
            log.error("STS authentication FAILED (%.1fs): %s", duration, e)
            raise RuntimeError(f"STS authentication failed: {e}") from e

        duration = time.time() - start
        self._token = token
        self._token_expires_at = time.time() + (self.token_cache_minutes * 60)
        log.info("STS authentication successful (%.1fs), token cached for %d minutes",
                 duration, self.token_cache_minutes)
        return token

    def lookup(self, id_in: str) -> str:
        """Look up the de-identified value for a given original identifier.

        This is a generic lookup - works for PatientID, PatientName, or any
        identifier. Mirrors HonestBrokerServiceImpl.lookup(idin).

        Returns the de-identified ID string (idOut), or raises on failure.
        Results are cached per input value for the lifetime of this client.
        """
        if id_in in self._id_cache:
            cached = self._id_cache[id_in]
            log.debug("Cache hit: %s -> %s", id_in, cached)
            return cached

        token = self.authenticate()

        encoded_id = urllib.parse.quote(id_in, safe="")
        api_url = f"https://{self.api_host}/DeIdentification/lookup?idIn={encoded_id}"

        log.debug("Calling HB API: GET %s", api_url)
        start = time.time()

        req = urllib.request.Request(
            api_url,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            duration = time.time() - start
            log.error("HB lookup FAILED (%.1fs) for idIn=%s: HTTP %d: %s",
                      duration, id_in, e.code, error_body)
            raise RuntimeError(
                f"HB lookup failed for '{id_in}': HTTP {e.code}: {error_body}"
            ) from e
        except Exception as e:
            duration = time.time() - start
            log.error("HB lookup FAILED (%.1fs) for idIn=%s: %s", duration, id_in, e)
            raise RuntimeError(f"HB lookup failed for '{id_in}': {e}") from e

        duration = time.time() - start

        results = json.loads(body)
        if not results or not isinstance(results, list) or len(results) == 0:
            log.error("HB lookup returned empty results (%.1fs) for idIn=%s", duration, id_in)
            raise RuntimeError(f"HB lookup returned no results for '{id_in}'")

        id_out = results[0].get("idOut")
        if not id_out:
            log.error("HB lookup result missing 'idOut' (%.1fs) for idIn=%s: %s",
                      duration, id_in, results[0])
            raise RuntimeError(f"HB lookup result missing 'idOut' for '{id_in}'")

        self._id_cache[id_in] = id_out
        log.info("HB lookup (%.1fs): %s -> %s", duration, id_in, id_out)
        return id_out


def find_dicom_files(input_dir: Path) -> list[Path]:
    """Recursively find all DICOM files in a directory.

    Attempts to identify DICOM files by extension (.dcm, .DCM) or by
    reading the DICOM preamble (files with no extension that start with
    the DICM magic bytes at offset 128).
    """
    dicom_files = []
    for root, _dirs, files in os.walk(input_dir):
        for filename in files:
            filepath = Path(root) / filename
            if filename.lower().endswith(".dcm"):
                dicom_files.append(filepath)
            elif "." not in filename or filename.lower().endswith(".ima"):
                # Try reading preamble for extensionless files
                try:
                    with open(filepath, "rb") as f:
                        f.seek(128)
                        magic = f.read(4)
                        if magic == b"DICM":
                            dicom_files.append(filepath)
                except (OSError, IOError):
                    pass
    dicom_files.sort()
    return dicom_files


def load_config(config_path: str) -> dict:
    """Load and validate the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    hb_config = config.get("honest_broker")
    if not hb_config:
        log.error("Config file missing 'honest_broker' section")
        sys.exit(1)

    required_keys = ["sts_host", "api_host", "app_name", "app_key", "username", "password"]
    missing = [k for k in required_keys if not hb_config.get(k)]
    if missing:
        log.error("Config file missing required keys: %s", ", ".join(missing))
        sys.exit(1)

    return hb_config


def relabel_directory(input_dir: Path, output_dir: Path, client: HonestBrokerClient,
                      dry_run: bool = False) -> dict:
    """Process all DICOM files in input_dir and write relabeled copies to output_dir.

    For each file:
      1. Reads PatientID (0010,0020) and PatientName (0010,0010)
      2. Calls HB lookup for each to get de-identified values
      3. Sets new PatientID = "{name_lookup}-{id_lookup}" (concatenated)
      4. Sets new PatientName = name_lookup value
      5. Writes to output preserving directory structure

    Returns a summary dict with counts, mappings, and any errors.
    """
    summary = {
        "total_files": 0,
        "processed": 0,
        "skipped": 0,
        "errors": [],
        "patient_id_mappings": {},
        "patient_name_mappings": {},
    }

    dicom_files = find_dicom_files(input_dir)
    summary["total_files"] = len(dicom_files)

    if not dicom_files:
        log.warning("No DICOM files found in %s", input_dir)
        return summary

    log.info("Found %d DICOM files in %s", len(dicom_files), input_dir)

    for filepath in dicom_files:
        relative_path = filepath.relative_to(input_dir)
        output_path = output_dir / relative_path

        try:
            ds = pydicom.dcmread(str(filepath), force=True)
        except Exception as e:
            log.error("Failed to read DICOM file %s: %s", filepath, e)
            summary["errors"].append({"file": str(relative_path), "error": str(e)})
            summary["skipped"] += 1
            continue

        original_patient_id = getattr(ds, "PatientID", None) or ""
        original_patient_name = str(getattr(ds, "PatientName", None) or "")

        if not original_patient_id:
            log.warning("No PatientID in %s, skipping", relative_path)
            summary["skipped"] += 1
            continue

        # Look up de-identified PatientID via HB
        try:
            hb_patient_id = client.lookup(original_patient_id)
        except RuntimeError as e:
            log.error("HB lookup failed for PatientID in %s: %s", relative_path, e)
            summary["errors"].append({
                "file": str(relative_path),
                "field": "PatientID",
                "original": original_patient_id,
                "error": str(e),
            })
            summary["skipped"] += 1
            continue

        # Look up de-identified PatientName
        new_patient_name = ""
        if original_patient_name:
            try:
                new_patient_name = client.lookup(original_patient_name)
            except RuntimeError as e:
                log.error("HB lookup failed for PatientName in %s: %s", relative_path, e)
                summary["errors"].append({
                    "file": str(relative_path),
                    "field": "PatientName",
                    "original": original_patient_name,
                    "error": str(e),
                })
                summary["skipped"] += 1
                continue

        # Combine: new PatientID = "{name_lookup}-{id_lookup}"
        # (matches MDA plugin's getExperimentLabel pattern)
        combined_patient_id = f"{new_patient_name}-{hb_patient_id}" if new_patient_name else hb_patient_id

        summary["patient_id_mappings"][original_patient_id] = combined_patient_id
        if original_patient_name:
            summary["patient_name_mappings"][original_patient_name] = new_patient_name

        # Show lookup details and changes
        log.info("  File: %s", relative_path)
        log.info("    PatientID   (0010,0020): lookup('%s') -> '%s'",
                 original_patient_id, hb_patient_id)
        if original_patient_name:
            log.info("    PatientName (0010,0010): lookup('%s') -> '%s'",
                     original_patient_name, new_patient_name)
        log.info("    Combined PatientID: '%s-%s' = '%s'",
                 new_patient_name, hb_patient_id, combined_patient_id)

        if dry_run:
            log.info("    [DRY RUN] No changes written")
            summary["processed"] += 1
            continue

        # Apply relabeling
        ds.PatientID = combined_patient_id
        ds.PatientName = new_patient_name

        log.info("    Changed PatientID:   '%s' -> '%s'", original_patient_id, combined_patient_id)
        log.info("    Changed PatientName: '%s' -> '%s'", original_patient_name, new_patient_name)

        # Write to output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ds.save_as(str(output_path))
        except Exception as e:
            log.error("Failed to write %s: %s", output_path, e)
            summary["errors"].append({"file": str(relative_path), "error": str(e)})
            summary["skipped"] += 1
            continue

        log.info("    Written to: %s", output_path)
        summary["processed"] += 1

    return summary


def print_summary(summary: dict) -> None:
    """Print a human-readable summary of the relabeling operation."""
    print("\n" + "=" * 60)
    print("HONEST BROKER RELABELING SUMMARY")
    print("=" * 60)
    print(f"Total DICOM files found: {summary['total_files']}")
    print(f"Successfully processed:  {summary['processed']}")
    print(f"Skipped/errors:          {summary['skipped']}")
    print()

    if summary["patient_id_mappings"]:
        print("PatientID Mappings (0010,0020):")
        for original, deidentified in sorted(summary["patient_id_mappings"].items()):
            print(f"  '{original}' -> '{deidentified}'")
        print()

    if summary["patient_name_mappings"]:
        print("PatientName Mappings (0010,0010):")
        for original, deidentified in sorted(summary["patient_name_mappings"].items()):
            print(f"  '{original}' -> '{deidentified}'")
        print()

    if summary["errors"]:
        print(f"Errors ({len(summary['errors'])}):")
        for err in summary["errors"]:
            parts = [err["file"]]
            if "field" in err:
                parts.append(f"{err['field']}='{err['original']}'")
            parts.append(err["error"])
            print(f"  {': '.join(parts)}")
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Relabel DICOM files using a remote MDA-style Honest Broker service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python honest_broker.py ../data/Elder_subject_florbetapir ./output
  python honest_broker.py ./input ./output --config my_config.yaml
  python honest_broker.py ./input ./output --dry-run
        """,
    )
    parser.add_argument("input_dir", help="Directory containing source DICOM files")
    parser.add_argument("output_dir", help="Directory for de-identified DICOM output")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without modifying any files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    if input_dir == output_dir:
        log.error("Input and output directories must be different")
        sys.exit(1)

    hb_config = load_config(args.config)
    client = HonestBrokerClient(hb_config)

    if args.dry_run:
        log.info("DRY RUN mode - no files will be written")

    log.info("Input:  %s", input_dir)
    log.info("Output: %s", output_dir)

    summary = relabel_directory(input_dir, output_dir, client, dry_run=args.dry_run)
    print_summary(summary)

    if summary["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
