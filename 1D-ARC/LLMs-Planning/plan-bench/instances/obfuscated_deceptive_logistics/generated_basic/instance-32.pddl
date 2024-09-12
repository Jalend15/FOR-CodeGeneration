
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Instance file automatically generated by the Tarski FSTRIPS writer
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem instance-32)
    (:domain obfuscated_deceptive_logistics)

    (:objects
        o0 o1 o10 o11 o12 o13 o14 o15 o2 o3 o4 o5 o6 o7 o8 o9 - object
    )

    (:init
        (cats o0)
        (cats o1)
        (stupendous o2)
        (stupendous o3)
        (sneeze o4)
        (sneeze o5)
        (texture o6)
        (texture o8)
        (texture o7)
        (texture o10)
        (texture o11)
        (texture o9)
        (collect o10 o3)
        (collect o7 o2)
        (collect o6 o2)
        (collect o8 o2)
        (collect o9 o3)
        (collect o11 o3)
        (spring o6)
        (spring o9)
        (hand o15)
        (hand o12)
        (hand o13)
        (hand o14)
        (next o14 o7)
        (next o5 o11)
        (next o13 o9)
        (next o0 o9)
        (next o12 o7)
        (next o4 o6)
        (next o1 o9)
        (next o15 o6)
    )

    (:goal
        (and (next o12 o11) (next o13 o7) (next o14 o11) (next o15 o8))
    )

    
    
    
)

