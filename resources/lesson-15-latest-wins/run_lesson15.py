#!/usr/bin/env python3
"""A CPU-only latest-wins and page-lifecycle simulation."""

import threading
import time


class Pages:
    def __init__(self):
        self.live = {1, 2, 3}  # persistent prefix
        self.next_page = 10

    def allocate(self):
        page = self.next_page
        self.next_page += 1
        self.live.add(page)
        return page

    def free(self, pages):
        self.live.difference_update(pages)


cancel = threading.Event()
pages = Pages()
result = {}


def old_generation():
    suffix = []
    try:
        generated = 0
        for step in range(20):
            if cancel.is_set():
                result['cancelled_at'] = step
                break
            suffix.append(pages.allocate())
            generated += 1
            time.sleep(0.005)
        result['generated'] = generated
    finally:
        pages.free(suffix)


worker = threading.Thread(target=old_generation)
worker.start()
time.sleep(0.018)
cancel.set()  # a newer keystroke arrives
worker.join()

print(result)
print('live pages after cancellation:', sorted(pages.live))
assert pages.live == {1, 2, 3}
