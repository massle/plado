class UnionFind:
    """
    Simple union find data structure.
    """

    def __init__(self, n: int = 0):
        self._uf: list[int] = list(range(n))
        self._classes: dict[int, list[int]] = {i: [i] for i in range(n)}

    def __str__(self):
        return "\n".join(
            (f"{i}: {self._const_lookup(i)}" for i in range(len(self._uf)))
        )

    def _const_lookup(self, i: int) -> int:
        assert i < len(self._uf)
        j = i
        k = self._uf[j]
        while j != k:
            j = k
            k = self._uf[j]
        return k

    def _lookup(self, i: int) -> int:
        k = self._const_lookup(i)
        while i != k:
            i, self._uf[i] = self._uf[i], k
        return i

    def resize(self, n: int) -> None:
        if n >= len(self._uf):
            self._classes.update({i: [i] for i in range(len(self._uf), n)})
            self._uf.extend(range(len(self._uf), n))
        else:
            self._uf = self._uf[:n]
            new_classes = {}
            for clas in self._classes.values():
                new_clas = [i for i in clas if i < n]
                if len(new_clas) > 0:
                    min_elem = min(new_clas)
                    new_classes[min_elem] = new_clas
            self._classes = new_classes

    def size(self) -> int:
        return len(self._uf)

    def merge(self, i: int, j: int) -> int:
        i, j = self._lookup(i), self._lookup(j)
        if i == j:
            return i
        i, j = (j, i) if j < i else (i, j)
        self._uf[j] = i
        self._classes[i].extend(self._classes[j])
        del self._classes[j]
        return i

    def are_same(self, i: int, j: int) -> None:
        return self._lookup(i) == self._lookup(j)

    def get_class(self, i: int) -> list[int]:
        return self._classes[self._lookup(i)]

    def __iter__(self):
        return iter(self._classes.values())

    def __getitem__(self, idx: int) -> int:
        return self._lookup(idx)
