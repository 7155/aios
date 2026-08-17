#!/usr/bin/env python3
"""Physical-page sharing simulation for Lesson 12."""

class Allocator:
    def __init__(self):
        self.next_page = 10
        self.live = set()

    def allocate(self, count):
        pages = list(range(self.next_page, self.next_page + count))
        self.next_page += count
        self.live.update(pages)
        return pages

    def free(self, pages):
        self.live.difference_update(pages)


alloc = Allocator()
prefix_len = 4
rows = 3
max_new_tokens = 4
prefix_pages = alloc.allocate(prefix_len)
page_table = [[None] * (prefix_len + max_new_tokens) for _ in range(rows)]
for row in page_table:
    row[:prefix_len] = prefix_pages

print("after one prefix prefill:")
for i, row in enumerate(page_table):
    print(i, row)
print("unique physical pages:", sorted(alloc.live))

suffix_pages = []
for row in range(rows):
    page = alloc.allocate(1)[0]
    suffix_pages.append(page)
    page_table[row][prefix_len] = page

print("\nafter one decode step:")
for i, row in enumerate(page_table):
    print(i, row)
print("unique physical pages:", sorted(alloc.live))

alloc.free(suffix_pages)
print("\nafter candidate text is materialized and suffix is freed:")
print("remaining physical pages:", sorted(alloc.live))
assert sorted(alloc.live) == prefix_pages
