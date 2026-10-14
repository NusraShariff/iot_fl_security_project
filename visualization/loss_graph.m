round = [1 2 3 4 5];
loss = [0.7683 0.7998 0.7794 0.7897 0.6519];

plot(round, loss, '-o', 'LineWidth', 2)
xlabel('Rounds')
ylabel('Log Loss')
title('Error vs Federated Learning Rounds')
grid on