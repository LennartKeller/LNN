import subprocess


def uconv_convert(
    text,
    from_code="UTF-8",
    to_code="UTF-8",
    transliterate: str = "Any-Latin",
    normalize: str = None,
):
    """
    Convert text encoding and apply Unicode transformations using uconv.

    Parameters:
    - text (str): The input string to be converted.
    - from_code (str): The source encoding (default is 'UTF-8').
    - to_code (str): The target encoding (default is 'UTF-8').
    - transliterate (str): Transliteration rule (optional).
    - normalize (str): Unicode normalization form (e.g., 'NFC', 'NFD', 'NFKC', 'NFKD') (optional).

    Returns:
    - str: The converted string.
    """
    cmd = ["uconv", "-f", from_code, "-t", to_code]

    if transliterate:
        cmd.extend(["-x", transliterate])

    if normalize:
        cmd.extend(["-x", f"any-{normalize.lower()}"])

    try:
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        input_bytes = text.encode(from_code)

        stdout, stderr = process.communicate(input=input_bytes)

        if process.returncode != 0:
            error_msg = stderr.decode(to_code)
            raise RuntimeError(f"uconv error: {error_msg}")

        output_text = stdout.decode(to_code)
        return output_text

    except FileNotFoundError:
        raise FileNotFoundError(
            "uconv command not found. Please ensure that ICU is installed and uconv is in your PATH."
        )
    except Exception as e:
        raise e
