"""Keep an Insta360 .insp photo's own data when its frame is cleaned.

WHAT AN .INSP FILE IS (checked on real files, 2026-11-09). A plain JPEG, with the
camera's own data in two places:

  * HEADER. Right after the JPEG start marker, before the picture data: the EXIF
    block plus a long run of 64 KB application blocks (about 4.9 MB on a 11904 x 5952
    frame). We do not know what they hold; they are not a color profile.
  * TRAILER. After the JPEG's end marker: about 6.2 MB. The last 40 bytes are a
    footer: the trailer's own length (4 bytes, little-endian), a version number
    (4 bytes) and a fixed 32-byte signature. The trailer holds a long stream of small
    numbers that looks like motion-sensor data. Searching it for the picture's byte
    size and for the whole file's size found neither: it does not point back at the
    picture's bytes, only at itself.

The cleaning engine rebuilds the picture and writes a fresh JPEG with only the EXIF
and color profile, so both pieces were being lost. Cleaning does not change the
picture's size or geometry, so this module copies the header blocks and the trailer
across, byte for byte, and names the result .insp.

Nothing here can touch the source: it only READS the original and WRITES the new
file. If anything about the source does not look like an .insp (no signature, an
impossible length, a trailer that does not start where the picture ends) the caller
gets False and keeps the plain .jpg it already wrote. No silent half-measures.
"""
import os
import struct

SIGNATURE = b"8db42d694ccc418790edff439fe026bf"
_FOOTER = 40                       # length field (4) + version (4) + signature (32)

# Blocks we never copy, because the writer already puts its own in the new file.
_EXIF = b"Exif\x00\x00"
_ICC = b"ICC_PROFILE\x00"
_JFIF = 0xE0


def _segments(f, limit):
    """Yield (marker, offset_of_marker, total_length_with_marker) for each JPEG
    segment from the start marker up to (not including) the picture data. `f` is an
    open binary file; `limit` stops a corrupt file from running past its end."""
    f.seek(2)
    pos = 2
    while pos + 4 <= limit:
        f.seek(pos)
        head = f.read(4)
        if len(head) < 4 or head[0] != 0xFF:
            return
        marker = head[1]
        if marker == 0xDA:                       # start of scan: the picture data
            return
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = struct.unpack(">H", head[2:4])[0]
        if length < 2:
            return
        yield marker, pos, 2 + length
        pos += 2 + length


def read_parts(path):
    """Return (header_blocks, trailer) from an Insta360 file, or None when the file
    does not have the structure described above.

    header_blocks is a list of complete segments (marker + length + payload), in
    their original order, leaving out the EXIF, color-profile and JFIF blocks."""
    try:
        size = os.path.getsize(path)
        if size < _FOOTER + 4:
            return None
        with open(path, "rb") as f:
            f.seek(size - _FOOTER)
            foot = f.read(_FOOTER)
            if foot[-32:] != SIGNATURE:
                return None
            trailer_len = struct.unpack("<I", foot[:4])[0]
            if not (_FOOTER <= trailer_len < size - 4):
                return None
            jpeg_end = size - trailer_len
            f.seek(0)
            if f.read(2) != b"\xff\xd8":
                return None
            f.seek(jpeg_end - 2)
            if f.read(2) != b"\xff\xd9":         # the picture must end where the trailer starts
                return None
            blocks = []
            for marker, off, total in _segments(f, jpeg_end):
                if not (0xE0 <= marker <= 0xEF):
                    continue
                f.seek(off)
                seg = f.read(total)
                payload = seg[4:]
                if marker == _JFIF:
                    continue
                if marker == 0xE1 and payload.startswith(_EXIF):
                    continue
                if marker == 0xE2 and payload.startswith(_ICC):
                    continue
                blocks.append(seg)
            f.seek(size - trailer_len)
            trailer = f.read(trailer_len)
            if len(trailer) != trailer_len:
                return None
        return blocks, trailer
    except (OSError, struct.error):
        return None


def _insert_point(data):
    """Byte offset in a freshly written JPEG just after its leading application
    blocks (JFIF, EXIF, color profile) -- where the carried blocks go."""
    pos = 2
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if not (0xE0 <= marker <= 0xEF):
            break
        pos += 2 + struct.unpack(">H", data[pos + 2:pos + 4])[0]
    return pos


def carry_insta360(src_path, new_jpeg_path, out_path):
    """Build `out_path` from the cleaned JPEG at `new_jpeg_path`, adding back the
    Insta360 header blocks and trailer found in `src_path`.

    Returns True when `out_path` was written (the plain JPEG at `new_jpeg_path` is
    then removed). Returns False and leaves the plain JPEG in place when the source
    is not a well-formed .insp or anything goes wrong."""
    parts = read_parts(src_path)
    if parts is None:
        return False
    blocks, trailer = parts
    tmp = out_path + ".part"
    try:
        with open(new_jpeg_path, "rb") as f:
            data = f.read()
        if data[:2] != b"\xff\xd8":
            return False
        at = _insert_point(data)
        carried = b"".join(blocks)
        result = data[:at] + carried + data[at:] + trailer
        if len(result) != len(data) + len(carried) + len(trailer):
            return False
        with open(tmp, "wb") as f:
            f.write(result)
        os.replace(tmp, out_path)
        if os.path.abspath(new_jpeg_path) != os.path.abspath(out_path):
            os.remove(new_jpeg_path)
        return True
    except OSError:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False
