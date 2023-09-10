import openai


parse_kg_prompt = """You are a mathematician and a scientist helping us extract relevant information from articles about mathematics. 
The task is to extract as many relevant relationships between entities to mathematics, physics, or history and science in general as possible. 
The entities should include all persons, mathematical entities, locations etc.
Specifically, the only entity tags you may use are:
Mathematical entity, Person, Location, Animal, Activity, Programming language, Equation, Date, Shape, Property, Mathematical expression, Profession, Time period, Mathematical subject, Mathematical concept, Discipline, Mathematical theorem, Physical entity, Physics subject, Physics. 
The only relationships you may use are: 
IS, ARE, WAS, EQUIVALENT_TO, CONTAINS, PROPOSED, PARTICIPATED_IN, SOLVED, RELATED_TO, CORRESPONDS_TO, HAS_PROPERTY, REPRESENTS, IS_USED_IN, DISCOVERED, FOUND, IS_SOLUTION_TO, PROVED, LIVED_IN, LIKED, BORN_IN, CONTRIBUTED_TO, IMPLIES, DESCRIBES, DEVELOPED, HAS_PROPERTY. USED_FOR.
As an example, if the text is "Euler was located in Sankt Petersburg in the 17 hundreds”, the output should have the following format Euler: Person, LIVED_IN,Skt. Petersburg: Location. If we have "In 1859, Rieman proved Theorem A”, then as an output you should return Riemann: Person, PROVED, Theorem A: Mathematical theorem.
I am only interested in the relationships in the above format and you can only use what you find in the text provided. Also, you should not provide relationships already found and you should choose less than 100 relationships and the most important ones.
You should only take the most important relationships as the aim is to build a knowledge graph. Rather a few but contextually meaningful than many nonsensical.
Moreover, you should only tag entities with one of the allowed tags if it truly fits that category and I am only interested in general entities such as “Shape HAS Area" rather than “Shape HAS Area 1”. 
The input text is the following: 
"""

test_passage = "In the late 1720s, Leonhard Euler was thinking about how to extend the factorial to non-integer values. This was the start of a rich theory used all over the scientific world. A theory of one of the most important functions in mathematics. Leonhard Euler is, without doubt, one of the greatest mathematicians in history. To give you an idea of Euler's powers, here are some examples that show his brilliance. First of all, Euler had an outstanding memory! He was able to recite Virgil's Aeneid from beginning to end, detailing in what line every page of the edition he owned began and ended. To give you some context, the Aeneid comprises a total of 9,896 lines! Euler was also extremely productive. He produced about 30,000 pages in his lifetime and it is estimated that he accounted for about a third of all published scientific papers in 18’th century!! He even published papers after he died! Many of those pages were written while he was blind, and for that reason, Euler has been called the Beethoven of mathematics. Beethoven could not hear his music. Likewise, Euler could not see his calculations. Actually, Euler was quite optimistic about the loss of his vision. He is known to have said something like: “In this way I will have fewer distractions”. One should think that this would slow him down, but in fact, when he became blind, his production went up! Euler also had phenomenal computational powers. On one occasion, two students disagreed over the result of the sum of 17 terms in a series because their results differed in the fifth decimal place. Euler computed the correct result in his mind in a few seconds. This anecdote was referred to by his colleague Nicolas de Condorcet who at Euler’s death wrote a lengthy eulogy in which he declares that Euler is “one of the greatest and most extraordinary men that nature has ever produced“. So Euler was a great mathematician, to say the least, and he was thinking about how to extend the factorial function. I will show you what he came up with and the surprising properties that followed. Later in the article, I'll reveal how we would give meaning to 1/2! and what the value of this symbol is."


def parse_to_knowledge_graph(article):
    print(f"article: {article[:100]}...")
    # model_name = "text-davinci-003"
    model_name = "gpt-4"

    completion = openai.Completion.create(
        model=model_name,
        temperature=0,
        # max_tokens=200,
        prompt=f"{parse_kg_prompt}{article}"
    )

    kg_result = completion['choices'][0]['text']
    return kg_result.replace('\n', '').strip()


if __name__ == "__main__":
    openai.api_key = ""
    kg_result = parse_to_knowledge_graph(test_passage)
    print(f"kg_result:\n{kg_result}")
