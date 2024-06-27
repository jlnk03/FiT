from iterators import ImageNetLatentIterator

if __name__ == "__main__":
     batch = next(iter(ImageNetLatentIterator({})))

     print(batch)