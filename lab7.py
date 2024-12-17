importrandom
 defdefine_problem():
 """Defineasampleoptimizationproblem:Spherefunction"""
 defsphere_function(genes):
 returnsum(x**2forxingenes)
 returnsphere_function
 definitialize_parameters():
 """SetparametersfortheGeneExpressionAlgorithm"""
 return{
 'population_size':50,
 'num_generations':100,
 'num_genes':5,
 'mutation_rate':0.1,
 'crossover_rate':0.8,
 'lower_bound':-10,
 'upper_bound':10,
 }
 definitialize_population(size,num_genes,lower_bound,upper_bound):
 """Generateaninitialrandompopulation"""
 return[[random.uniform(lower_bound,upper_bound)for_in
 range(num_genes)]for_inrange(size)]
 defevaluate_fitness(population,fitness_function):
 """Evaluatefitnessofeachindividual"""
 return[fitness_function(individual)forindividualinpopulation]
 defselection(population,fitness,num_parents):
 """Selectindividualsbasedonfitness(tournamentselection)"""
 selected=[]
 for_inrange(num_parents):
 candidates=random.sample(list(zip(population,fitness)),k=3)
 selected.append(min(candidates,key=lambdax:x[1])[0])
 returnselected
 defcrossover(parent1,parent2,crossover_rate):
 """Performcrossoverbetweentwoparents"""
 ifrandom.random()<crossover_rate:
 point=random.randint(1,len(parent1)-1)
returnparent1[:point]+parent2[point:],parent2[:point]+
 parent1[point:]
 returnparent1,parent2
 defmutate(individual,mutation_rate,lower_bound,upper_bound):
 """Mutateanindividualbymodifyinggenes"""
 return[
 gene+random.uniform(-1,1)ifrandom.random()<mutation_rateelse
 gene
 forgeneinindividual
 ]
 defgene_expression_algorithm():
 """MainfunctiontoexecutetheGeneExpressionAlgorithm"""
 fitness_function=define_problem()
 params=initialize_parameters()
 population=initialize_population(
 params['population_size'],
 params['num_genes'],
 params['lower_bound'],
 params['upper_bound']
 )
 forgenerationinrange(params['num_generations']):
 fitness=evaluate_fitness(population,fitness_function)
 parents=selection(population,fitness,params['population_size']//
 2)
 next_generation=[]
 foriinrange(0,len(parents),2):
 p1,p2=parents[i],parents[min(i+1,len(parents)-1)]
 offspring1,offspring2=crossover(p1,p2,
 params['crossover_rate'])
 next_generation.extend([
 mutate(offspring1,params['mutation_rate'],
 params['lower_bound'],params['upper_bound']),
 mutate(offspring2,params['mutation_rate'],
 params['lower_bound'],params['upper_bound']),
 ])
 population=next_generation
best_fitness = min(fitness)
 print(f"Generation {generation}: Best Fitness = {best_fitness}")
 final_fitness = evaluate_fitness(population, fitness_function)
 best_solution = population[final_fitness.index(min(final_fitness))]
 print("Final Best Solution:", best_solution)
 gene_expression_algorithm()