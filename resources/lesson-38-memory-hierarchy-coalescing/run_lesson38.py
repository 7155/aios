def segments(stride_words, lanes=32, word_bytes=4, segment_bytes=32):
    addresses=[lane*stride_words*word_bytes for lane in range(lanes)]
    touched=sorted({a//segment_bytes for a in addresses})
    return addresses, touched

for stride in (1,2,8,32):
    addresses,touched=segments(stride)
    print(f'stride={stride:2}: touched segments={len(touched):2}, first addresses={addresses[:8]}')

print('\nShared bank map (32 banks):')
for stride in (1,32,33):
    banks=[(lane*stride)%32 for lane in range(32)]
    print(f'stride={stride}: unique banks={len(set(banks))}, first={banks[:8]}')
