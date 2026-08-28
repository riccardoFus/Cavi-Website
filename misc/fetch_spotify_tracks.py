#!/usr/bin/env python3
"""
Fetch all track IDs from ALL Spotify playlists (47) using Spotify API.
Requires Client Credentials (set in environment or hardcoded).

Usage: python3 fetch_spotify_tracks.py
Output: spotify_tracks.json
"""

import json
import urllib.request
import urllib.error
import base64
import os
import time
import sys

# Spotify Client Credentials
CLIENT_ID = "9e8dc331b590429b80c278ed8dd00c1a"
CLIENT_SECRET = "d272fe7b4cf04c1bbc3b829000359a24"
SPOTIFY_USER_ID = "11134519888"

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "spotify_tracks.json")


def get_access_token():
    """Get Spotify API access token via Client Credentials flow."""
    auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=b"grant_type=client_credentials",
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data["access_token"]


def api_get(url, token, retries=3):
    """Make a GET request to Spotify API with retry on rate limit."""
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", 2)) + 1
                print(f"    ⏳ Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise Exception(f"Failed after {retries} retries: {url}")


def fetch_all_playlists(token):
    """Fetch all playlists for the user with pagination."""
    all_playlists = []
    offset = 0
    while True:
        url = f"https://api.spotify.com/v1/users/{SPOTIFY_USER_ID}/playlists?limit=50&offset={offset}"
        data = api_get(url, token)
        items = data.get("items", [])
        all_playlists.extend(items)
        total = data.get("total", 0)
        print(f"  📋 Fetched {len(items)} playlists ({len(all_playlists)}/{total})")
        if len(all_playlists) >= total or not items:
            break
        offset += 50
    return all_playlists


def fetch_playlist_tracks(playlist_id, playlist_name, token):
    """Fetch all track IDs from a playlist with pagination."""
    tracks = []
    offset = 0
    while True:
        url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit=100&offset={offset}&fields=items(track(id)),total"
        try:
            data = api_get(url, token)
        except Exception as e:
            print(f"    ⚠️  Error fetching tracks for {playlist_name}: {e}", file=sys.stderr)
            break

        items = data.get("items", [])
        for item in items:
            track = item.get("track")
            if track and track.get("id"):
                tracks.append({"id": track["id"], "playlist": playlist_name})

        total = data.get("total", 0)
        if len(tracks) >= total or not items:
            break
        offset += 100
        time.sleep(0.1)  # Small delay to be nice to the API

    return tracks


def main():
    print("🎵 Spotify Track Fetcher — cavi.wav (ALL 47 playlists)")
    print("=" * 60)

    # Step 1: Get access token
    print("\n1️⃣  Authenticating with Spotify API...")
    token = get_access_token()
    print(f"   ✅ Token acquired: {token[:25]}...")

    # Step 2: Fetch all playlists
    print("\n2️⃣  Fetching all playlists...")
    playlists = fetch_all_playlists(token)
    print(f"   ✅ Found {len(playlists)} playlists")

    # Step 3: Fetch tracks from each playlist
    print(f"\n3️⃣  Fetching tracks from all playlists...")
    all_tracks = []
    for i, pl in enumerate(playlists, 1):
        name = pl["name"]
        pl_id = pl["id"]
        track_count = pl.get("tracks", {}).get("total", 0)
        print(f"   [{i:2d}/{len(playlists)}] {name[:40]:40s} ({track_count:5d} tracks)...", end=" ", flush=True)

        tracks = fetch_playlist_tracks(pl_id, name, token)
        all_tracks.extend(tracks)
        print(f"✅ {len(tracks)} fetched")

    # Step 4: Deduplicate by track ID
    print(f"\n4️⃣  Deduplicating...")
    seen_ids = set()
    unique_tracks = []
    for t in all_tracks:
        if t["id"] not in seen_ids:
            seen_ids.add(t["id"])
            unique_tracks.append(t)

    # Step 5: Save to JSON
    output = {
        "user": "cavi.wav",
        "spotify_user_id": SPOTIFY_USER_ID,
        "total_tracks": len(unique_tracks),
        "total_playlists": len(playlists),
        "playlists": [pl["name"] for pl in playlists],
        "tracks": unique_tracks,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"📊 Results:")
    print(f"   Playlists processed: {len(playlists)}")
    print(f"   Total tracks found:  {len(all_tracks)}")
    print(f"   Unique tracks:       {len(unique_tracks)}")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print(f"✅ Done!")


if __name__ == "__main__":
    main()