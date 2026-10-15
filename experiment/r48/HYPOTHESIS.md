# r48 — BPE to the ceiling, and past the char cap

r47: BPE single-token keys give 49.8% cold at 8 facts (char tax gone, char==exact). The 50->96 gap is the
bootstrapping axis (r25). Two questions.

- **H48a — curriculum takes BPE 8 facts to the ceiling.** 4 facts cold (should be near the 96% ceiling, as char's 4
  facts were near 90%), then 8 facts warm from it. Predicted: well above r47's cold 49.8%, approaching 96%. If it
  stalls near 50%, the bootstrapping escape does not transfer to BPE and something else caps it.
- **H48b — BPE scales past the char 13-fact cap.** The char task maxes at 13 facts (2 letters/key from 26); BPE has
  3799 single-token words, so 16 and 32 facts are available. Cold baselines; if they hold well above chance
  (~1/1899), the fact axis is no longer vocabulary-bound.

box1 = ladder, box2 = scale.
