def validate_pdf(uploaded_file):
    """Check whether a file was uploaded and is a PDF."""

    if uploaded_file is None:
        return False

    if uploaded_file.type != "application/pdf":
        return False

    return True