(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i h j k c a l g f)
(:init 
(handempty)
(ontable i)
(ontable h)
(ontable j)
(ontable k)
(ontable c)
(ontable a)
(ontable l)
(ontable g)
(ontable f)
(clear i)
(clear h)
(clear j)
(clear k)
(clear c)
(clear a)
(clear l)
(clear g)
(clear f)
)
(:goal
(and
(on i h)
(on h j)
(on j k)
(on k c)
(on c a)
(on a l)
(on l g)
(on g f)
)))