assert functools.partial(<function downsize at 0x7fdbde803c40>, newShape=(1, 33))(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), newShape=(1, 33)) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function downsizeMode at 0x7fdbde805a80>, newShape=(1, 33))(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), newShape=(1, 33)) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function countColors at 0x7fdbde805620>, rotate=-1, outBackgroundColor=-1, flip=True, sliced=False, ignore='max', outShape='inShape', byShape=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), rotate=-1, outBackgroundColor=-1, flip=True, sliced=False, ignore='max', outShape='inShape', byShape=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function countShapes at 0x7fdbde805760>, color=-1, outShape='inShape', lay='h', outColor=None, shape=None, skip=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), color=-1, outShape='inShape', lay='h', outColor=None, shape=None, skip=True) == [[0 0 8 0 0 0 8 0 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function colorMap at 0x7fdbde8014e0>, cMap={3: 0, 8: 0, 6: 0})(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), cMap={3: 0, 8: 0, 6: 0}) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function symmetrize at 0x7fdbde7c3740>, axis=['ud'], color=8)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), axis=['ud'], color=8) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function identityM at 0x7fdbde99f880>)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function deletePixels at 0x7fdbde802e80>, diagonals=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), diagonals=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function deletePixels at 0x7fdbde802e80>, diagonals=False)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), diagonals=False) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function predictCNN at 0x7fdbde7c3c40>, model=OneConvModel(
  (conv): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), bias=False)
  (pad): ConstantPad2d(padding=(1, 1, 1, 1), value=0)
), commonColors=[0], nChannels=2)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), model=OneConvModel(
  (conv): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), bias=False)
  (pad): ConstantPad2d(padding=(1, 1, 1, 1), value=0)
), commonColors=[0], nChannels=2) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function moveAllShapes at 0x7fdbde802ac0>, color=[8], background=-1, direction='any', until=-2, nSteps=4)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), color=[8], background=-1, direction='any', until=-2, nSteps=4) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function doPixelMod2Row at 0x7fdbde801ee0>, rules={(0, 1): 0})(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), rules={(0, 1): 0}) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function identityM at 0x7fdbde99f880>)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function identityM at 0x7fdbde99f880>)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function identityM at 0x7fdbde99f880>)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function paintShapeFromBorderColor at 0x7fdbde801800>, shapeColors=set(), fixedColors=set(), diagonals=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), shapeColors=set(), fixedColors=set(), diagonals=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, connColor=0)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), connColor=0) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, connColor=0, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), connColor=0, diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=0, connColor=0, fixedColors=set())(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, connColor=0, fixedColors=set()) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=0, connColor=0, fixedColors=set(), diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, connColor=0, fixedColors=set(), diagonal=True) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=3, connColor=0, fixedColors=set())(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, connColor=0, fixedColors=set()) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=3, connColor=0, fixedColors=set(), diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, connColor=0, fixedColors=set(), diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=4, connColor=0, fixedColors=set())(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, connColor=0, fixedColors=set()) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=4, connColor=0, fixedColors=set(), diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, connColor=0, fixedColors=set(), diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=6, connColor=0, fixedColors=set())(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, connColor=0, fixedColors=set()) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=6, connColor=0, fixedColors=set(), diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, connColor=0, fixedColors=set(), diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=8, connColor=0, fixedColors=set())(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=8, connColor=0, fixedColors=set()) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=8, connColor=0, fixedColors=set(), diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=8, connColor=0, fixedColors=set(), diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=0, allowedChanges={3: 0, 8: 0, 6: 0})(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, allowedChanges={3: 0, 8: 0, 6: 0}) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=0, allowedChanges={3: 0, 8: 0, 6: 0}, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, allowedChanges={3: 0, 8: 0, 6: 0}, diagonal=True) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=0, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=0, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=0, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True, diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=3, allowedChanges={3: 0, 8: 0, 6: 0})(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, allowedChanges={3: 0, 8: 0, 6: 0}) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=3, allowedChanges={3: 0, 8: 0, 6: 0}, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, allowedChanges={3: 0, 8: 0, 6: 0}, diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=3, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=3, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=3, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True, diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=4, allowedChanges={3: 0, 8: 0, 6: 0})(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, allowedChanges={3: 0, 8: 0, 6: 0}) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=4, allowedChanges={3: 0, 8: 0, 6: 0}, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, allowedChanges={3: 0, 8: 0, 6: 0}, diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=4, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=4, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=4, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True, diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=6, allowedChanges={3: 0, 8: 0, 6: 0})(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, allowedChanges={3: 0, 8: 0, 6: 0}) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=6, allowedChanges={3: 0, 8: 0, 6: 0}, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, allowedChanges={3: 0, 8: 0, 6: 0}, diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=6, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=6, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=6, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True, diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=8, allowedChanges={3: 0, 8: 0, 6: 0})(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=8, allowedChanges={3: 0, 8: 0, 6: 0}) == [[0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=8, allowedChanges={3: 0, 8: 0, 6: 0}, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=8, allowedChanges={3: 0, 8: 0, 6: 0}, diagonal=True) == [[0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=8, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=8, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function connectAnyPixels at 0x7fdbde802fc0>, pixelColor=8, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True, diagonal=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), pixelColor=8, allowedChanges={3: 0, 8: 0, 6: 0}, lineExclusive=True, diagonal=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function alignShapes at 0x7fdbde806200>, refColor=0)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), refColor=0) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function symmetrizeSubmatrix at 0x7fdbde8013a0>, lr=True, ud=True, rotation=False, newColor=None)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), lr=True, ud=True, rotation=False, newColor=None) == [[8 0 0 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function replicateShapes at 0x7fdbde8063e0>, diagonal=True, multicolor=True, deleteOriginal=False, anchorType='subframe', allCombs=False, attributes={'MoCl'})(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), diagonal=True, multicolor=True, deleteOriginal=False, anchorType='subframe', allCombs=False, attributes={'MoCl'}) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function symmetrizeAllShapes at 0x7fdbde8058a0>, targetColor=0)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), targetColor=0) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function colorByPixels at 0x7fdbde805bc0>, deletePixels=False)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), deletePixels=False) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function layShapes at 0x7fdbde8060c0>, firstPos=(1, 1), direction=(-1, -1), diagonal=True, multicolor=True, outShape='inShape', overlap=(0, 0), sortBy='grid', reverse=False)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), firstPos=(1, 1), direction=(-1, -1), diagonal=True, multicolor=True, outShape='inShape', overlap=(0, 0), sortBy='grid', reverse=False) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function replicateOneShape at 0x7fdbde806520>, lay='pixelwise', multicolor=False, paintLikePix=True)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), lay='pixelwise', multicolor=False, paintLikePix=True) == [[0 0 0 8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function paintGridLikeBackground at 0x7fdbde805940>)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
assert functools.partial(<function cropAllBackground at 0x7fdbde806d40>)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[8 8 8 8 8 8 8 8 8 8 8 0 0 8 0 0 8 0 0 0 8]]
assert functools.partial(<function switchColors at 0x7fdbde803420>)(array([[0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 8, 0, 0, 8, 0, 0,
        0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]])) == [[8 8 8 0 0 0 0 0 0 0 0 0 0 0 8 8 0 8 8 0 8 8 8 0 8 8 8 8 8 8 8 8 8]]
