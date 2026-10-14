rounds = [1 2 3 4 5];

baseline = [0.88 0.88 0.88 0.88 0.88];
basic_fl = [0.70 0.72 0.73 0.74 0.74];
fl_trust = [0.80 0.85 0.88 0.90 0.91];
final_model = [0.876 0.906 0.863 0.846 0.895];

plot(rounds, baseline, '-o', 'LineWidth', 2)
hold on
plot(rounds, basic_fl, '-s', 'LineWidth', 2)
plot(rounds, fl_trust, '-^', 'LineWidth', 2)
plot(rounds, final_model, '-d', 'LineWidth', 2)

xlabel('Rounds')
ylabel('Accuracy')
title('Model Performance over Federated Learning Rounds')

legend('Baseline','Basic FL','FL + Trust','Final Model')

grid on