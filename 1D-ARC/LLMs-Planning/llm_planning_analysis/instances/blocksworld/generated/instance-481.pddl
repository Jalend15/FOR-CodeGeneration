(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d g k c b l i j f h a)
(:init 
(handempty)
(ontable d)
(ontable g)
(ontable k)
(ontable c)
(ontable b)
(ontable l)
(ontable i)
(ontable j)
(ontable f)
(ontable h)
(ontable a)
(clear d)
(clear g)
(clear k)
(clear c)
(clear b)
(clear l)
(clear i)
(clear j)
(clear f)
(clear h)
(clear a)
)
(:goal
(and
(on d g)
(on g k)
(on k c)
(on c b)
(on b l)
(on l i)
(on i j)
(on j f)
(on f h)
(on h a)
)))