assert downsize(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), newShape=(1, 33)) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert downsizeMode(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), newShape=(1, 33)) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert countColors(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), rotate=-1, outBackgroundColor=-1, flip=True, sliced=False, ignore='max', outShape='inShape', byShape=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert countShapes(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), color=-1, outShape='inShape', lay='h', outColor=None, shape=None, skip=True) == [[0 0 'a' 0 0 0 'a' 0 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert colorMap(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), cMap={3: 0, 'a': 0, 6: 0}) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert symmetrize(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), axis=['ud'], color='a') == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert identityM(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert deletePixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), diagonals=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert deletePixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), diagonals=False) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert predictCNN(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), model=OneConvModel(
  (conv): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), bias=False)
  (pad): ConstantPad2d(padding=(1, 1, 1, 1), value=0)
), commonColors=[0], nChannels=2) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert moveAllShapes(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), color=['a'], background=-1, direction='any', until=-2, nSteps=4) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert doPixelMod2Row(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), rules={(0, 1): 0}) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert identityM(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert identityM(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert identityM(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert paintShapeFromBorderColor(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), shapeColors=set(), fixedColors=set(), diagonals=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), connColor=0) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), connColor=0, diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, connColor=0, fixedColors=set()) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, connColor=0, fixedColors=set(), diagonal=True) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, connColor=0, fixedColors=set()) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, connColor=0, fixedColors=set(), diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, connColor=0, fixedColors=set()) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, connColor=0, fixedColors=set(), diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, connColor=0, fixedColors=set()) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, connColor=0, fixedColors=set(), diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor='a', connColor=0, fixedColors=set()) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor='a', connColor=0, fixedColors=set(), diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, allowedChanges={3: 0, 'a': 0, 6: 0}) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, allowedChanges={3: 0, 'a': 0, 6: 0}, diagonal=True) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, allowedChanges={3: 0, 'a': 0, 6: 0}, lineExclusive=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, allowedChanges={3: 0, 'a': 0, 6: 0}, lineExclusive=True, diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, allowedChanges={3: 0, 'a': 0, 6: 0}) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, allowedChanges={3: 0, 'a': 0, 6: 0}, diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, allowedChanges={3: 0, 'a': 0, 6: 0}, lineExclusive=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, allowedChanges={3: 0, 'a': 0, 6: 0}, lineExclusive=True, diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, allowedChanges={3: 0, 'a': 0, 6: 0}) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, allowedChanges={3: 0, 'a': 0, 6: 0}, diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, allowedChanges={3: 0, 'a': 0, 6: 0}, lineExclusive=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, allowedChanges={3: 0, 'a': 0, 6: 0}, lineExclusive=True, diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, allowedChanges={3: 0, 'a': 0, 6: 0}) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, allowedChanges={3: 0, 'a': 0, 6: 0}, diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, allowedChanges={3: 0, 'a': 0, 6: 0}, lineExclusive=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, allowedChanges={3: 0, 'a': 0, 6: 0}, lineExclusive=True, diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor='a', allowedChanges={3: 0, 'a': 0, 6: 0}) == [[0 0 0 'a' 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor='a', allowedChanges={3: 0, 'a': 0, 6: 0}, diagonal=True) == [[0 0 0 'a' 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor='a', allowedChanges={3: 0, 'a': 0, 6: 0}, lineExclusive=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert connectAnyPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor='a', allowedChanges={3: 0, 'a': 0, 6: 0}, lineExclusive=True, diagonal=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert alignShapes(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), refColor=0) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert symmetrizeSubmatrix(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), lr=True, ud=True, rotation=False, newColor=None) == [['a' 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert replicateShapes(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), diagonal=True, multicolor=True, deleteOriginal=False, anchorType='subframe', allCombs=False, attributes={'MoCl'}) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert symmetrizeAllShapes(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), targetColor=0) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert colorByPixels(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), deletePixels=False) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert layShapes(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), firstPos=(1, 1), direction=(-1, -1), diagonal=True, multicolor=True, outShape='inShape', overlap=(0, 0), sortBy='grid', reverse=False) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert replicateOneShape(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]]), lay='pixelwise', multicolor=False, paintLikePix=True) == [[0 0 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a' 0 0 0 0 0 0 0 0 0]]
assert paintGridLikeBackground(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert cropAllBackground(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [['a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 0 0 'a' 0 0 'a' 0 0 0 'a']]
assert switchColors(array([[0, 0, 0, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 0, 0, 'a', 0, 0, 'a', 0, 0,
        0, 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [['a' 'a' 'a' 0 0 0 0 0 0 0 0 0 0 0 'a' 'a' 0 'a' 'a' 0 'a' 'a' 'a' 0 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a']]