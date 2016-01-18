# TODO

# TODO 20160107

- To study the need of END token

  - END is not mandatory to maintain valid parsings
    - END should be delivered by explicit control
    - Pre-mature pausing of parsing process allows more flexibility
    - Simplifies representation of grammar
      - For grammars without lookahead, END can totally be ignored
      - For grammars with lookahead, END can be borrowed temporarily when constructing the parsing table