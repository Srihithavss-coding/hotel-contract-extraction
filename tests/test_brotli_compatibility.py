from httpx2._decoders import BrotliDecoder, brotli


def test_httpx2_brotli_decoder_supports_output_buffer_limit():
    assert brotli is not None

    original = b"Offline HTTPX2 Brotli decoder compatibility check."
    compressed = brotli.compress(original)

    assert b"".join(BrotliDecoder().decode(compressed)) == original
