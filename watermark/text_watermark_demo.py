import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LogitsProcessorList

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

device = "cuda"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.5,
                                               delta=2.0,
                                               seeding_scheme="simple_1") #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
# Note:
# You can turn off self-hashing by setting the seeding scheme to `minhash`.

original_text = "This month some important Walmart news did not get the attention it deserved. The news wasn't some flashy announcement about virtual reality or about some new service for busy Manhattanites. It was something much more subtle and much more substantive.\nIt was the hire of Valerie Casey. Google \"Valerie Casey\" even today and only two listings make mention of her newfound connection with Walmart on the first page of the returned search results. This amount of attention is unjustifiably slight. Let this piece then serve as the hire's coming out party.\nShrewdly, Walmart has made Valerie Casey, formerly of Frog, Ideo, and Pentagram, its new head of design. She will be charged with leading Walmart into the future and ensuring that all its products and services, from websites to employee and consumer apps, will work in cohesion and from a singular experience design point of view.\n\"What's the big whup?\" you might ask.\nThe big whup is that this announcement clearly shows that Walmart has jumped feet first into the waters of next generation retailing. Under Doug McMillon's bold leadership, Walmart here again is taking one of the most essential steps to long-term success for any legacy bricks-and-mortar retailer—specifically, redefining how it thinks of the word \"product.\"\nInstead of doing what many retailers do, like staffing de facto \"product\" heads within owned brand development, store operations, e-commerce, etc. and then leaving them to collaborate inside complex organizations with competing priorities, Walmart rightly understands that retailing has become so complex that all these aforementioned roles now need their own steward to ensure alignment.\nWalmart having a mission to help its customers save money and live better is one thing, but putting that promise into coherent action is another thing entirely. Many retailers take such coordinated action for granted, which is why so many legacy retail experiences feel disjointed across the digital and physical divide. Success rarely is found in the compromise and turf wars of internal politics. The buck needs to stop somewhere.\nPrior to e-commerce, before customers had the myriad of options they have today to consume products (via web portals, apps, social media, etc.), retailers were able to differentiate themselves by the products inside their four walls. The strategic practice of product differentiation became so prevalent and such a point of emphasis that it even gave way to the commonly heard parlance of retail being a \"merchant-led business.\"\nThe growth of e-commerce in the 1990s and 2000s rewrote the playbook though. During this period, Jeff Bezos built Amazon into the \"Everything Store.\" His idea was that as long as he could offer a great selection, convenience and great prices, consumers would gravitate towards Amazon over time. Boy, was he right, and, boy, did they ever gravitate towards Amazon.\nWhether customers try to order direct from Amazon or from one of Amazon's many third-party suppliers on its marketplace, it is becoming more and more difficult for a consumer not to find what he or she desires on Amazon. At this point, nearly 50% of all first product searches start on Amazon, indicating that Amazon is settling into the digital and product acquisition-equivalent of the 1980s shopping mall. The products inside a retailers four walls or within a retailers assortment are now, in a way, almost non-differentiating.\nBezos saw that the opportunity in retail was to put the customer first and to think not about the product he or she buys but about the Product he or she really buys—for him, the Product (big \"P) of the Amazon brand. The white space in retail was no longer leading with product (small \"p\"), but leading with the discipline of Product Management, where his Product, or in this case, the Amazon brand, was the magic created at the intersection of great experience design, technology and business. Product was not a pair of khaki pants, a new set of bed sheets or a bar of soap. It was the culmination of a friction-free user experience that delivered on the universal truths of selection, convenience, and low prices and that exceeded consumers' expectations consistently, again and again.\nSadly, many retailers have not grasped this \"Product Problem.\" The greatest Product Manager in history, Jeff Bezos, saw the problem early on, while many merchant princes missed the signals and have even now admitted that they underestimated the impact technology would have on retail. Fortunately, Walmart's recent hire of Casey indicates that there could be light at the end of what has, until now, been a dark tunnel.\nWalmart is big. The acquisitions of Jet.com, Bonobos, ModCloth, etc. have only made it bigger and more complex, saying nothing too of the ever-changing demands of consumers and Walmart's ever-expanding interests abroad (see Flipkart). Therefore, it is only right that someone, like Casey, take up the mantle to oversee and to fight for the consumer and to ensure that all Walmart's activities look, feel, and convey the Walmart brand in the simplest, most straightforward way as possible.\nThe products inside Walmart's stores are immaterial to its future success. What matters is the Product of its brand—its website, its store, its app-based services, etc. Those are the Products that matter. In a future world where the only thing that differentiates a physical from a digital experience is the memory and delight of being somewhere, a Walmart store is the Product or collective set of experiences that will get someone off his or her couch.\nA tube of toothpaste just won't cut it anymore.\nWhile the products within Walmart's store or on its website will come and go and ebb and flow, the shroud of the Walmart brand will be what matters. Walmart's store and its digital properties will be the envelopes that carry its brand promise.\nThe experiment that bears this out has already been run. We need look no further than the history of Amazon and also to the young upstart CEOs like Emily Weiss, Katrina Lake and Adam Goldenberg who are already thriving in this new world. Their collective early success indicates that they get the punchline to the joke of 21st century retail—that the retail of the future won't be merchant led, it will be Product led.\nBest of luck, Valerie Casey. A bricks-and-mortar nation turns its lonely eyes to you."
input_words_n = 30
input_text = " ".join(original_text.split()[:input_words_n])

tokenized_input = tokenizer(input_text, return_tensors="pt").to(model.device)
# note that if the model is on cuda, then the input is on cuda
# and thus the watermarking rng is cuda-based.
# This is a different generator than the cpu-based rng in pytorch!

generation_config = {
                "max_length": 200 + 10,
                # "no_repeat_ngram_size": 4,
                "logits_processor": LogitsProcessorList([watermark_processor])
            }

output_tokens = model.generate(**tokenized_input,
                               **generation_config)

# if decoder only model, then we need to isolate the
# newly generated tokens as only those are watermarked, the input/prompt is not
output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
print(f"Output:\n{output_text}")
print(f"{'='*10}")


# detect
watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.5, # should match original setting
                                        seeding_scheme="simple_1", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        # ignore_repeated_ngrams=True
                                        )

score_dict = watermark_detector.detect(output_text) # or any other text of interest to analyze
print(score_dict)
print(f"{'='*10}")


score_dict = watermark_detector.detect(original_text[:len(output_text)]) # or any other text of interest to analyze
print(score_dict)
print(f"{'='*10}")
