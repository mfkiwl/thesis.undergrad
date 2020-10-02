"""
test for huffman coding
"""
def main():
    from huffman import HuffmanCoding
    import sys

    inputFilePath = "sample.txt"
    handle = HuffmanCoding(inputFilePath)
    output_path = handle.compress()
    print("Compressed file path: " + output_path)
    decom_path = h.decompress(output_path)
    print("Decompressed file path: " + decom_path)


if __name__ == "__main__":
    main()
